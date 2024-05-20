use crate::binary::{instruction::Instruction, types::{FuncType, ValueType}};
use anyhow::{bail, Result};

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Function {
    pub locals: Vec<ValueType>,
    pub body: Vec<Instruction>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InternalFuncInst {
    pub fn_type: FuncType,
    pub code: Function,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FuncInst {
    Internal(InternalFuncInst),
}

#[derive(Default)]
pub struct Store {
    pub fns: Vec<FuncInst>,
}
impl Store {
    pub fn new(module: crate::binary::module::Module) -> Result<Self> {
        let decded_fns = match module.fn_section {
            Some(ref func_types) => func_types.clone(),
            _ => vec![],
        };

        let mut fns = vec![];

        if let Some(section) = module.code_section {
            for (body, i) in section.iter().zip(decded_fns.into_iter()) {
                let Some(ref func_types) = module.type_section else {
                    bail!("Not found type section")
                };

                let Some(fn_type) = func_types.get(i as usize) else {
                    bail!("Not found func type in section")
                };

                let locals = body.locals.iter()
                    .flat_map(|lc| std::iter::repeat(lc.value_type.clone()).take(lc.type_count as usize))
                    .collect::<Vec<_>>()
                ;

                fns.push(FuncInst::Internal(InternalFuncInst { 
                    fn_type: fn_type.clone(), 
                    code: Function { locals, body: body.code.clone() }
                }));
            }
        }

        Ok(Self { fns })
    }
}