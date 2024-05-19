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
    use crate::{binary::{instruction::Instruction, module::Module, types::{FuncType, ValueType}}, execution::{runtime::Runtime, store::{FuncInst, Function, InternalFuncInst, Store}, value::Value}};

    #[test]
    fn execute_fn_add() -> Result<()> {
        let wasm = wat::parse_str("(module (func (param i32 i32)(result i32) (local.get 0) (local.get 1) i32.add))")?;
        let mut instance = Runtime::instanciate(wasm)?;

        assert_eq!(Some(Value::I32(5)), instance.call(0, vec![Value::I32(2), Value::I32(3)])?);
        assert_eq!(Some(Value::I32(15)), instance.call(0, vec![Value::I32(10), Value::I32(5)])?);
        assert_eq!(Some(Value::I32(2)), instance.call(0, vec![Value::I32(1), Value::I32(1)])?);
        Ok(())
    }

    #[test]
    fn init_store() -> Result<()> {
        let wasm = wat::parse_str("(module (func (param i32 i32)(result i32) (local.get 0) (local.get 1) i32.add))")?;
        let module = Module::new(&wasm)?;
        let store = Store::new(module)?;

        assert_eq!(1, store.fns.len());

        let expect = InternalFuncInst {
                fn_type: FuncType { params: vec![ValueType::I32, ValueType::I32], returns: vec![ValueType::I32] },
                code: Function { 
                    locals: vec![], 
                    body: vec![
                        Instruction::LocalGet(0),
                        Instruction::LocalGet(1),
                        Instruction::I32Add,
                        Instruction::End,
                    ] 
                },
            }
        ;
        assert_eq!(FuncInst::Internal(expect), store.fns[0]);
        Ok(())
    }
}