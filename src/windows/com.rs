use std::mem::MaybeUninit;
use std::os::windows::prelude::*;
use std::time::Duration;
use std::{io, ptr};

use winapi::um::ioapiset::CancelIo;

use winapi::shared::minwindef::*;
use winapi::shared::winerror::{ERROR_IO_PENDING, WAIT_TIMEOUT};
use winapi::um::commapi::*;
use winapi::um::errhandlingapi::GetLastError;
use winapi::um::fileapi::*;
use winapi::um::handleapi::*;
use winapi::um::processthreadsapi::GetCurrentProcess;
use winapi::um::winbase::*;
use winapi::um::winnt::{
    DUPLICATE_SAME_ACCESS, FILE_ATTRIBUTE_NORMAL, GENERIC_READ, GENERIC_WRITE, HANDLE, MAXDWORD,
};

use winapi::um::ioapiset::GetOverlappedResult;
use winapi::um::minwinbase::OVERLAPPED;
use winapi::um::synchapi::{CreateEventW, WaitForSingleObject};
use winapi::um::winbase::{INFINITE, WAIT_OBJECT_0};

use crate::windows::dcb;
use crate::{
    ClearBuffer, DataBits, Error, ErrorKind, FlowControl, Parity, Result, SerialPort,
    SerialPortBuilder, StopBits,
};

struct Overlapped(OVERLAPPED);

impl Overlapped {
    #[inline]
    fn new() -> io::Result<Self> {
        let event = unsafe { CreateEventW(ptr::null_mut(), TRUE, FALSE, ptr::null_mut()) };
        if event.is_null() {
            return Err(io::Error::last_os_error());
        }
        Ok(Self(OVERLAPPED {
            Internal: 0,
            InternalHigh: 0,
            u: unsafe { std::mem::zeroed() },
            hEvent: event,
        }))
    }
}

impl Drop for Overlapped {
    #[inline(always)]
    fn drop(&mut self) {
        unsafe {
            CloseHandle(self.0.hEvent);
        }
    }
}

/// A serial port implementation for Windows COM ports
///
/// The port will be closed when the value is dropped. However, this struct
/// should not be instantiated directly by using `COMPort::open()`, instead use
/// the cross-platform `serialport::open()` or
/// `serialport::open_with_settings()`.
#[derive(Debug)]
pub struct COMPort {
    handle: HANDLE,
    timeout: Duration,
    port_name: Option<String>,
}

unsafe impl Send for COMPort {}

impl COMPort {
    /// Opens a COM port as a serial device.
    ///
    /// `port` should be the name of a COM port, e.g., `COM1`.
    ///
    /// If the COM port handle needs to be opened with special flags, use
    /// `from_raw_handle` method to create the `COMPort`. Note that you should
    /// set the different settings before using the serial port using `set_all`.
    ///
    /// ## Errors
    ///
    /// * `NoDevice` if the device could not be opened. This could indicate that
    ///    the device is already in use.
    /// * `InvalidInput` if `port` is not a valid device name.
    /// * `Io` for any other I/O error while opening or initializing the device.
    pub fn open(builder: &SerialPortBuilder) -> Result<COMPort> {
        let mut name = Vec::<u16>::with_capacity(4 + builder.path.len() + 1);

        name.extend(r"\\.\".encode_utf16());
        name.extend(builder.path.encode_utf16());
        name.push(0);

        let handle = unsafe {
            CreateFileW(
                name.as_ptr(),
                GENERIC_READ | GENERIC_WRITE,
                0,
                ptr::null_mut(),
                OPEN_EXISTING,
                FILE_ATTRIBUTE_NORMAL | FILE_FLAG_OVERLAPPED,
                0 as HANDLE,
            )
        };

        if handle == INVALID_HANDLE_VALUE {
            return Err(super::error::last_os_error());
        }

        // create the COMPort here so the handle is getting closed
        // if one of the calls to `get_dcb()` or `set_dcb()` fails
        let mut com = COMPort::open_from_raw_handle(handle as RawHandle);

        let mut dcb = dcb::get_dcb(handle)?;
        dcb::init(&mut dcb);
        dcb::set_baud_rate(&mut dcb, builder.baud_rate);
        dcb::set_data_bits(&mut dcb, builder.data_bits);
        dcb::set_parity(&mut dcb, builder.parity);
        dcb::set_stop_bits(&mut dcb, builder.stop_bits);
        dcb::set_flow_control(&mut dcb, builder.flow_control);
        dcb::set_dcb(handle, dcb)?;

        // Try to set DTR on best-effort.
        if let Some(dtr) = builder.dtr_on_open {
            let _ = com.write_data_terminal_ready(dtr);
        }

        com.set_timeout(builder.timeout)?;
        com.port_name = Some(builder.path.clone());
        Ok(com)
    }

    /// Attempts to clone the `SerialPort`. This allow you to write and read simultaneously from the
    /// same serial connection. Please note that if you want a real asynchronous serial port you
    /// should look at [mio-serial](https://crates.io/crates/mio-serial) or
    /// [tokio-serial](https://crates.io/crates/tokio-serial).
    ///
    /// Also, you must be very careful when changing the settings of a cloned `SerialPort` : since
    /// the settings are cached on a per object basis, trying to modify them from two different
    /// objects can cause some nasty behavior.
    ///
    /// This is the same as `SerialPort::try_clone()` but returns the concrete type instead.
    ///
    /// # Errors
    ///
    /// This function returns an error if the serial port couldn't be cloned.
    pub fn try_clone_native(&self) -> Result<COMPort> {
        let process_handle: HANDLE = unsafe { GetCurrentProcess() };
        let mut cloned_handle: HANDLE = INVALID_HANDLE_VALUE;
        unsafe {
            DuplicateHandle(
                process_handle,
                self.handle,
                process_handle,
                &mut cloned_handle,
                0,
                TRUE,
                DUPLICATE_SAME_ACCESS,
            );
            if cloned_handle != INVALID_HANDLE_VALUE {
                Ok(COMPort {
                    handle: cloned_handle,
                    port_name: self.port_name.clone(),
                    timeout: self.timeout,
                })
            } else {
                Err(super::error::last_os_error())
            }
        }
    }

    fn escape_comm_function(&mut self, function: DWORD) -> Result<()> {
        match unsafe { EscapeCommFunction(self.handle, function) } {
            0 => Err(super::error::last_os_error()),
            _ => Ok(()),
        }
    }

    fn read_pin(&mut self, pin: DWORD) -> Result<bool> {
        let mut status: DWORD = 0;

        match unsafe { GetCommModemStatus(self.handle, &mut status) } {
            0 => Err(super::error::last_os_error()),
            _ => Ok(status & pin != 0),
        }
    }

    fn open_from_raw_handle(handle: RawHandle) -> Self {
        // It is not trivial to get the file path corresponding to a handle.
        // We'll punt and set it `None` here.
        COMPort {
            handle: handle as HANDLE,
            timeout: Duration::from_millis(100),
            port_name: None,
        }
    }

    fn timeout_constant(duration: Duration) -> DWORD {
        let milliseconds = duration.as_millis();
        // In the way we are setting up COMMTIMEOUTS, a timeout_constant of MAXDWORD gets rejected.
        // Let's clamp the timeout constant for values of MAXDWORD and above. See remarks at
        // https://learn.microsoft.com/en-us/windows/win32/api/winbase/ns-winbase-commtimeouts.
        //
        // This effectively throws away accuracy for really long timeouts but at least preserves a
        // long-ish timeout. But just casting to DWORD would result in presumably unexpected short
        // and non-monotonic timeouts from cutting off the higher bits.
        u128::min(milliseconds, MAXDWORD as u128 - 1) as DWORD
    }
}

impl Drop for COMPort {
    fn drop(&mut self) {
        unsafe {
            CloseHandle(self.handle);
        }
    }
}

impl AsRawHandle for COMPort {
    fn as_raw_handle(&self) -> RawHandle {
        self.handle as RawHandle
    }
}

impl FromRawHandle for COMPort {
    unsafe fn from_raw_handle(handle: RawHandle) -> Self {
        COMPort::open_from_raw_handle(handle)
    }
}

impl IntoRawHandle for COMPort {
    fn into_raw_handle(self) -> RawHandle {
        let Self { handle, .. } = self;

        handle as RawHandle
    }
}

impl io::Read for COMPort {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let mut len: DWORD = 0;

        let mut overlapped = Overlapped::new()?;

        let process_result = |len| -> io::Result<usize> {
            if len != 0 {
                return Ok(len as usize);
            }
            // if timeout occured len == 0
            Err(io::Error::new(
                io::ErrorKind::TimedOut,
                "Operation timed out",
            ))
        };

        match unsafe {
            ReadFile(
                self.handle,
                buf.as_mut_ptr() as LPVOID,
                buf.len() as DWORD,
                &mut len,
                &mut overlapped.0,
            )
        } {
            FALSE => {
                let result_len = wait_overlapped_result(self.timeout, self.handle, overlapped)?;
                return process_result(result_len);
            }
            _ => process_result(len),
        }
    }
}

impl io::Write for COMPort {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let mut len: DWORD = 0;

        let mut overlapped = Overlapped::new()?;

        match unsafe {
            WriteFile(
                self.handle,
                buf.as_ptr() as LPVOID,
                buf.len() as DWORD,
                &mut len,
                &mut overlapped.0,
            )
        } {
            FALSE => {
                let result_len = wait_overlapped_result(self.timeout, self.handle, overlapped)?;
                return Ok(result_len as usize);
            }
            _ => Ok(len as usize),
        }
    }

    fn flush(&mut self) -> io::Result<()> {
        match unsafe { FlushFileBuffers(self.handle) } {
            0 => Err(io::Error::last_os_error()),
            _ => Ok(()),
        }
    }
}

fn wait_overlapped_result(
    timeout: Duration,
    handle: HANDLE,
    mut overlapped: Overlapped,
) -> io::Result<u32> {
    match unsafe { GetLastError() } {
        ERROR_IO_PENDING => {
            let timeout = u128::min(timeout.as_millis(), INFINITE as u128 - 1) as u32;
            let mut len: DWORD = 0;
            match unsafe { WaitForSingleObject(overlapped.0.hEvent, timeout) } as u32 {
                WAIT_OBJECT_0 => {
                    if unsafe { GetOverlappedResult(handle, &mut overlapped.0, &mut len, TRUE) }
                        == TRUE
                    {
                        return Ok(len);
                    }
                    Err(io::Error::last_os_error())
                }
                WAIT_TIMEOUT => {
                    cancel_io(handle, &mut overlapped);
                    Err(io::Error::new(
                        io::ErrorKind::TimedOut,
                        "Operation timed out",
                    ))
                }
                _ => Err(io::Error::last_os_error()),
            }
        }
        _ => Err(io::Error::last_os_error()),
    }
}

fn cancel_io(handle: HANDLE, overlapped: &mut Overlapped) {
    if unsafe { CancelIoEx(handle, &mut overlapped.0) } == TRUE {
        let mut len = 0;
        let _ = unsafe { GetOverlappedResult(handle, &mut overlapped.0, &mut len, TRUE) };
    }
}

impl SerialPort for COMPort {
    fn name(&self) -> Option<String> {
        self.port_name.clone()
    }

    fn timeout(&self) -> Duration {
        self.timeout
    }

    fn set_timeout(&mut self, timeout: Duration) -> Result<()> {
        //let timeout_constant = Self::timeout_constant(timeout);

        let mut timeouts = COMMTIMEOUTS {
            ReadIntervalTimeout: 0,
            ReadTotalTimeoutMultiplier: 0,
            ReadTotalTimeoutConstant: 0,
            WriteTotalTimeoutMultiplier: 0,
            WriteTotalTimeoutConstant: 0,
        };

        if unsafe { SetCommTimeouts(self.handle, &mut timeouts) } == 0 {
            return Err(super::error::last_os_error());
        }

        self.timeout = timeout;
        Ok(())
    }

    fn write_request_to_send(&mut self, level: bool) -> Result<()> {
        if level {
            self.escape_comm_function(SETRTS)
        } else {
            self.escape_comm_function(CLRRTS)
        }
    }

    fn write_data_terminal_ready(&mut self, level: bool) -> Result<()> {
        if level {
            self.escape_comm_function(SETDTR)
        } else {
            self.escape_comm_function(CLRDTR)
        }
    }

    fn read_clear_to_send(&mut self) -> Result<bool> {
        self.read_pin(MS_CTS_ON)
    }

    fn read_data_set_ready(&mut self) -> Result<bool> {
        self.read_pin(MS_DSR_ON)
    }

    fn read_ring_indicator(&mut self) -> Result<bool> {
        self.read_pin(MS_RING_ON)
    }

    fn read_carrier_detect(&mut self) -> Result<bool> {
        self.read_pin(MS_RLSD_ON)
    }

    fn baud_rate(&self) -> Result<u32> {
        let dcb = dcb::get_dcb(self.handle)?;
        Ok(dcb.BaudRate as u32)
    }

    fn data_bits(&self) -> Result<DataBits> {
        let dcb = dcb::get_dcb(self.handle)?;
        match dcb.ByteSize {
            5 => Ok(DataBits::Five),
            6 => Ok(DataBits::Six),
            7 => Ok(DataBits::Seven),
            8 => Ok(DataBits::Eight),
            _ => Err(Error::new(
                ErrorKind::Unknown,
                "Invalid data bits setting encountered",
            )),
        }
    }

    fn parity(&self) -> Result<Parity> {
        let dcb = dcb::get_dcb(self.handle)?;
        match dcb.Parity {
            ODDPARITY => Ok(Parity::Odd),
            EVENPARITY => Ok(Parity::Even),
            NOPARITY => Ok(Parity::None),
            _ => Err(Error::new(
                ErrorKind::Unknown,
                "Invalid parity bits setting encountered",
            )),
        }
    }

    fn stop_bits(&self) -> Result<StopBits> {
        let dcb = dcb::get_dcb(self.handle)?;
        match dcb.StopBits {
            TWOSTOPBITS => Ok(StopBits::Two),
            ONESTOPBIT => Ok(StopBits::One),
            _ => Err(Error::new(
                ErrorKind::Unknown,
                "Invalid stop bits setting encountered",
            )),
        }
    }

    fn flow_control(&self) -> Result<FlowControl> {
        let dcb = dcb::get_dcb(self.handle)?;
        if dcb.fOutxCtsFlow() != 0 || dcb.fRtsControl() != 0 {
            Ok(FlowControl::Hardware)
        } else if dcb.fOutX() != 0 || dcb.fInX() != 0 {
            Ok(FlowControl::Software)
        } else {
            Ok(FlowControl::None)
        }
    }

    fn set_baud_rate(&mut self, baud_rate: u32) -> Result<()> {
        let mut dcb = dcb::get_dcb(self.handle)?;
        dcb::set_baud_rate(&mut dcb, baud_rate);
        dcb::set_dcb(self.handle, dcb)
    }

    fn set_data_bits(&mut self, data_bits: DataBits) -> Result<()> {
        let mut dcb = dcb::get_dcb(self.handle)?;
        dcb::set_data_bits(&mut dcb, data_bits);
        dcb::set_dcb(self.handle, dcb)
    }

    fn set_parity(&mut self, parity: Parity) -> Result<()> {
        let mut dcb = dcb::get_dcb(self.handle)?;
        dcb::set_parity(&mut dcb, parity);
        dcb::set_dcb(self.handle, dcb)
    }

    fn set_stop_bits(&mut self, stop_bits: StopBits) -> Result<()> {
        let mut dcb = dcb::get_dcb(self.handle)?;
        dcb::set_stop_bits(&mut dcb, stop_bits);
        dcb::set_dcb(self.handle, dcb)
    }

    fn set_flow_control(&mut self, flow_control: FlowControl) -> Result<()> {
        let mut dcb = dcb::get_dcb(self.handle)?;
        dcb::set_flow_control(&mut dcb, flow_control);
        dcb::set_dcb(self.handle, dcb)
    }

    fn bytes_to_read(&self) -> Result<u32> {
        let mut errors: DWORD = 0;
        let mut comstat = MaybeUninit::uninit();

        if unsafe { ClearCommError(self.handle, &mut errors, comstat.as_mut_ptr()) != 0 } {
            unsafe { Ok(comstat.assume_init().cbInQue) }
        } else {
            Err(super::error::last_os_error())
        }
    }

    fn bytes_to_write(&self) -> Result<u32> {
        let mut errors: DWORD = 0;
        let mut comstat = MaybeUninit::uninit();

        if unsafe { ClearCommError(self.handle, &mut errors, comstat.as_mut_ptr()) != 0 } {
            unsafe { Ok(comstat.assume_init().cbOutQue) }
        } else {
            Err(super::error::last_os_error())
        }
    }

    fn clear(&self, buffer_to_clear: ClearBuffer) -> Result<()> {
        let buffer_flags = match buffer_to_clear {
            ClearBuffer::Input => PURGE_RXABORT | PURGE_RXCLEAR,
            ClearBuffer::Output => PURGE_TXABORT | PURGE_TXCLEAR,
            ClearBuffer::All => PURGE_RXABORT | PURGE_RXCLEAR | PURGE_TXABORT | PURGE_TXCLEAR,
        };

        if unsafe { PurgeComm(self.handle, buffer_flags) != 0 } {
            Ok(())
        } else {
            Err(super::error::last_os_error())
        }
    }

    fn try_clone(&self) -> Result<Box<dyn SerialPort>> {
        match self.try_clone_native() {
            Ok(p) => Ok(Box::new(p)),
            Err(e) => Err(e),
        }
    }

    fn set_break(&self) -> Result<()> {
        if unsafe { SetCommBreak(self.handle) != 0 } {
            Ok(())
        } else {
            Err(super::error::last_os_error())
        }
    }

    fn clear_break(&self) -> Result<()> {
        if unsafe { ClearCommBreak(self.handle) != 0 } {
            Ok(())
        } else {
            Err(super::error::last_os_error())
        }
    }
}

use winapi::um::ioapiset::CancelIoEx;
use winapi::um::synchapi::SetEvent;
use winapi::um::synchapi::WaitForMultipleObjects;

use futures::task::AtomicWaker;
use std::collections::VecDeque;
use std::sync::Arc;
use std::sync::Mutex;
use std::task::{Context, Poll};

#[derive(Debug, Clone)]
struct HandleWrapper(HANDLE);

unsafe impl Send for HandleWrapper {}
unsafe impl Sync for HandleWrapper {}

#[derive(Debug, Clone, Copy)]
pub enum BackPressure {
    Cache,
    DropOldest { threshold: usize },
}

///
#[derive(Debug)]
pub struct WindowsRxEventStream {
    rx_thread: Option<std::thread::JoinHandle<()>>,
    abort_event: HandleWrapper,
    chunk_size: Option<usize>,
    inner: Arc<EventsInner>,
}

#[derive(Debug)]
struct EventsInner {
    events: Mutex<(Vec<u8>, Option<Error>)>,
    waker: AtomicWaker,
}

impl WindowsRxEventStream {
    ///
    pub fn new(
        port: &COMPort,
        chunk_size: Option<usize>,
        back_pressure: BackPressure,
    ) -> Result<Self> {
        let handle = HandleWrapper(port.handle);

        // enable EV_RXCHAR
        if unsafe { SetCommMask(handle.0, 1) } == FALSE {
            return Err(io::Error::last_os_error().into());
        }

        let abort_event: HANDLE =
            unsafe { CreateEventW(ptr::null_mut(), TRUE, FALSE, ptr::null_mut()) };
        if abort_event.is_null() {
            return Err(io::Error::last_os_error().into());
        }
        let abort_event = HandleWrapper(abort_event);
        let file_handle_cloned = handle.clone();
        let abort_event_cloned = abort_event.clone();

        let inner = Arc::new(EventsInner {
            events: Mutex::new((Vec::new(), None)),
            waker: AtomicWaker::new(),
        });
        let events_cloned = inner.clone();
        let rx_thread = Some(std::thread::spawn(move || {
            rx_events_process(file_handle_cloned, abort_event_cloned, events_cloned);
        }));
        Ok(Self {
            rx_thread,
            abort_event,
            chunk_size,
            inner,
        })
    }

    ///
    pub fn try_poll_next(&mut self, cx: &mut Context) -> Poll<Option<Result<Vec<u8>>>> {
        if let Some(error) = &self.inner.events.lock().unwrap().1 {
            return Poll::Ready(Some(Err(error.clone())));
        }

        self.inner.waker.register(cx.waker());

        let queue = &mut self.inner.events.lock().unwrap().0;
        let len = self.chunk_size.unwrap_or(queue.len());
        if len > 0 && queue.len() >= len {
            let buffer: Vec<_> = queue.drain(..len).collect();

            if queue.len() > 0 {
                println!("back pressure: {}", queue.len());
            }
            return Poll::Ready(Some(Ok(buffer)));
        }

        return Poll::Pending;
    }
}

impl Drop for WindowsRxEventStream {
    fn drop(&mut self) {
        if unsafe { SetEvent(self.abort_event.0) } == TRUE {
            self.rx_thread.take().unwrap().join().unwrap();
        }
        unsafe { CloseHandle(self.abort_event.0) };
    }
}

fn rx_events_process(
    file_handle: HandleWrapper,
    abort_event: HandleWrapper,
    inner: Arc<EventsInner>,
) {
    if let Err(e) = rx_loop_event(file_handle, abort_event, inner.clone()) {
        inner.events.lock().unwrap().1 = Some(e);
        inner.waker.wake();
    }
}

fn rx_loop_event(
    file_handle: HandleWrapper,
    abort_event: HandleWrapper,
    inner: Arc<EventsInner>,
) -> Result<()> {
    // purge all input buffer cached data first

    send_pending_data(&file_handle, &inner)?;

    loop {
        match unsafe { WaitForSingleObject(abort_event.0, 0) } {
            0 => {
                // aborted already
                return Ok(());
            }
            WAIT_TIMEOUT => {
                let mut overlapped = Overlapped::new()?;
                let mut mask = 0;
                if unsafe { WaitCommEvent(file_handle.0, &mut mask, &mut overlapped.0) } == FALSE {
                    if unsafe { GetLastError() } == ERROR_IO_PENDING {
                        let objects = [overlapped.0.hEvent, abort_event.0];
                        const WAIT_OBJECT_1: u32 = WAIT_OBJECT_0 + 1;
                        match unsafe {
                            WaitForMultipleObjects(
                                objects.len() as u32,
                                objects.as_ptr(),
                                FALSE,
                                INFINITE,
                            )
                        } {
                            WAIT_OBJECT_0 => {
                                let mut len = 0;
                                if unsafe {
                                    GetOverlappedResult(
                                        file_handle.0,
                                        &mut overlapped.0,
                                        &mut len,
                                        TRUE,
                                    )
                                } == TRUE
                                {
                                    send_pending_data(&file_handle, &inner)?;
                                } else {
                                    return Err(io::Error::last_os_error().into());
                                }
                            }
                            WAIT_OBJECT_1 => {
                                cancel_io(file_handle.0, &mut overlapped);

                                // abort signaled just return
                                return Ok(());
                            }
                            _ => {
                                return Err(io::Error::last_os_error().into());
                            }
                        }
                    } else {
                        return Err(io::Error::last_os_error().into());
                    }
                } else {
                    return Err(io::Error::last_os_error().into());
                }
            }
            _ => {
                return Err(io::Error::last_os_error().into());
            }
        }
    }
}

fn send_pending_data(file_handle: &HandleWrapper, inner: &Arc<EventsInner>) -> Result<()> {
    let mut errors: DWORD = 0;
    let mut comstat = MaybeUninit::uninit();

    if unsafe { ClearCommError(file_handle.0, &mut errors, comstat.as_mut_ptr()) == TRUE } {
        let mut len = unsafe { comstat.assume_init() }.cbInQue;
        if len > 0 {
            let mut buf: Vec<u8> = vec![0; len as usize];
            let mut overlapped_read = Overlapped::new()?;

            match unsafe {
                ReadFile(
                    file_handle.0,
                    buf.as_mut_ptr() as LPVOID,
                    buf.len() as DWORD,
                    &mut len,
                    &mut overlapped_read.0,
                )
            } {
                TRUE => {
                    let queue = &mut inner.events.lock().unwrap().0;
                    queue.extend(buf);
                    inner.waker.wake();
                    return Ok(());
                }
                _ => {
                    return Err(io::Error::new(
                        io::ErrorKind::Other,
                        "ReadFile returned false for cached data",
                    )
                    .into());
                }
            }
        } else {
            return Ok(());
        }
    } else {
        return Err(io::Error::last_os_error().into());
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tests::timeout::MONOTONIC_DURATIONS;

    #[test]
    fn timeout_constant_is_monotonic() {
        let mut last = COMPort::timeout_constant(Duration::ZERO);

        for (i, d) in MONOTONIC_DURATIONS.iter().enumerate() {
            let next = COMPort::timeout_constant(*d);
            assert!(
                next >= last,
                "{next} >= {last} failed for {d:?} at index {i}"
            );
            last = next;
        }
    }

    #[test]
    fn timeout_constant_zero_is_zero() {
        assert_eq!(0, COMPort::timeout_constant(Duration::ZERO));
    }
}
