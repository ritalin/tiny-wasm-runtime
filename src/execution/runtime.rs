use anyhow::Result;

use super::value::Value;

#[derive(Default)]
pub struct Runtime {

}
impl Runtime {
    pub fn instanciate(wasm: Vec<u8>) -> Result<Self> {
        todo!()
    }

    pub fn call(&mut self, fn_index: usize, args: Vec<Value>) -> Result<Option<Value>> {
        todo!()
    }
}

#[cfg(test)]
mod executor_tests {
    use anyhow::Result;
    use crate::execution::{runtime::Runtime, value::Value};

    #[test]
    fn execute_fn_add() -> Result<()> {
        let wasm = wat::parse_str("(module (func (param i32 i32)(result i32) (local.get 0) (local.get 1) i32.add))")?;
        let mut instance = Runtime::instanciate(wasm)?;

        assert_eq!(Some(Value::I32(5)), instance.call(0, vec![Value::I32(2), Value::I32(3)])?);
        assert_eq!(Some(Value::I32(15)), instance.call(0, vec![Value::I32(10), Value::I32(5)])?);
        assert_eq!(Some(Value::I32(2)), instance.call(0, vec![Value::I32(1), Value::I32(1)])?);
        Ok(())
    }
}