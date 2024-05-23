
use std::{fs::File, io::Write, os::fd::{FromRawFd, IntoRawFd}};
use anyhow::{bail, Result};

use super::{store::Store, value::Value};

#[derive(Debug)]
pub struct WasiSnapshotPreview1 {
}

impl WasiSnapshotPreview1 {
    pub fn new() -> Self {
        Self {}
    }
}

impl WasiSnapshotPreview1 {
    pub fn invoke(&self, store: &mut Store, fn_name: &str, args: Vec<Value>) -> Result<Option<Value>> {
        let result = match fn_name {
            "fd_write" => {
                let len = fd_write(&mut SyscallContext { store }, args[0].into(), args[1].into(), args[2].into(), args[3].into())?;
                Some(Value::I32(len))
            }
            _ => unimplemented!(),
        };

        Ok(result)
    }
}

struct SyscallContext<'a> {
    store: &'a mut Store,
}

fn fd_write(ctx: &mut SyscallContext, fd: i32, iovs: i32, iovs_len: i32, rp: i32) -> Result<i32> {
    let Some(memory) = ctx.store.memories.get_mut(0) else {
        bail!("memory page is not found");
    };

    let rp = rp as usize;
    let mut iovs = iovs as usize;
    let mut write_len = 0;

    for _ in 0..iovs_len {
        let offset = read_i32(&memory.data, iovs)? as usize;
        iovs += 4;
        let len = read_i32(&memory.data, iovs)? as usize;

        let bytes = &memory.data[offset..][..len];

        write_len += match try_write_to_stdio(fd, bytes)? {
            Some(len) => len,
            None => write_to_file(fd, bytes)?,
        }
    }

    memory.data[rp..][..4].copy_from_slice(&write_len.to_le_bytes());

    Ok(write_len)
}

fn read_i32(memory: &[u8], offset: usize) -> Result<i32> {
    Ok(<i32>::from_le_bytes(memory[offset..][..std::mem::size_of::<i32>()].try_into()?))
}
    
fn try_write_to_stdio(fd: i32, bytes: &[u8]) -> Result<Option<i32>> {
    let fmt = String::from_utf8_lossy(bytes);
    match fd {
        0 => panic!("can not write stdin"),
        1 => print!("{fmt}"),
        2 => eprint!("{fmt}"),
        _ => return Ok(None),
    };

    Ok(Some(bytes.len() as i32))
}

fn write_to_file(fd: i32, bytes: &[u8]) -> Result<i32> {
    let mut file = unsafe { File::from_raw_fd(fd) };
    file.write_all(bytes)?;
    let _ = file.into_raw_fd();

    Ok(bytes.len() as i32)
}

#[cfg(test)]
mod syscall_test {
    use std::{io::Seek, os::fd::{FromRawFd, IntoRawFd}};

    use capture_stdio::{Capture, OutputCapture};
    use anyhow::Result;
    use tempfile::tempfile;

    use crate::execution::{store::{MemoryInst, Store}, wasi::SyscallContext};

    fn init_memory_with_str(memory: &mut [u8], s: &str) {
        let mut offset: usize = 0;
        let mut len = s.len();
        memory[offset..][..len].copy_from_slice(s.as_bytes());

        offset += len;
        len = std::mem::size_of::<i32>();
        memory[offset..][..len].copy_from_slice(&0i32.to_le_bytes());

        offset += len;
        memory[offset..][..len].copy_from_slice(&(s.len() as i32).to_le_bytes());
    }

    #[test]
    fn syscall_write_to_stdout() -> Result<()> {
        let mut store = Store { 
            memories: vec![
                MemoryInst { data: vec![0; 100], limit: None },
            ],
            ..Default::default()
        };

        const EXPECTED: &str = "Hello!Hello!";
        if let Some(memory) = store.memories.get_mut(0) {
            init_memory_with_str(&mut memory.data, EXPECTED);
        }
        
        let (len, output) = {
            let capture = OutputCapture::capture()?;
            let mut ctx = SyscallContext { store: &mut store };

            let len = super::fd_write(&mut ctx, 1, EXPECTED.len() as i32, 1, (EXPECTED.len() + 8) as i32)?;

            let output = String::from_utf8(capture.get_output().lock().unwrap().clone()).unwrap();

            (len, output)
        };

        assert_eq!(EXPECTED, output);
        assert_eq!(EXPECTED.len(), len as usize);

        Ok(())
    }

    #[test]
    fn syscall_write_to_file() -> Result<()> {
        let mut store = Store { 
            memories: vec![
                MemoryInst { data: vec![0; 100], limit: None },
            ],
            ..Default::default()
        };

        const EXPECTED: &str = "Hello!Hello!";
        if let Some(memory) = store.memories.get_mut(0) {
            init_memory_with_str(&mut memory.data, EXPECTED);
        }

        let (len, output) = {
            let file = tempfile()?;
            let fd = file.into_raw_fd();

            let mut ctx = SyscallContext { store: &mut store };
            let len = super::fd_write(&mut ctx, fd, EXPECTED.len() as i32, 1, (EXPECTED.len() + 8) as i32)?;
            
            let mut file = unsafe { std::fs::File::from_raw_fd(fd) };
            let mut output = String::new();

            use std::io::Read;
            file.seek(std::io::SeekFrom::Start(0))?;
            file.read_to_string(&mut output)?;

            (len, output)
        };

        assert_eq!(EXPECTED.len(), len as usize);
        assert_eq!(EXPECTED, &output);

        Ok(())
    }
}