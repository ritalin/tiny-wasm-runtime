use std::collections::HashMap;

use crate::binary::{instruction::Instruction, types::{Export, FuncType, ValueType}};
use anyhow::{bail, Result};

pub const PPAGE_SIZE: usize = (1 << 16) - 1;

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
pub struct ExternalFuncInst {
    pub fn_type: FuncType,
    pub mod_name: String,
    pub fn_name: String,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FuncInst {
    Internal(InternalFuncInst),
    External(ExternalFuncInst),
}

#[derive(Debug, Default)]
pub struct ExportContainer {
    pub lookup: HashMap<String, Export>,
}
#[derive(Debug, Default)]
pub struct MemoryInst {
    pub data: Vec<u8>,
    pub limit: Option<u32>,
}

#[derive(Debug, Default)]
pub struct Store {
    pub fns: Vec<FuncInst>,
    pub exports: ExportContainer,
    pub memories: Vec<MemoryInst>,
}
impl Store {
    pub fn new(module: crate::binary::module::Module) -> Result<Self> {
        let decded_fns = match module.fn_section {
            Some(ref func_types) => func_types.clone(),
            _ => vec![],
        };

        let type_section_count = match &module.type_section {
            Some(section) => section.len(),
            None => 0,
        };

        let mut fns = Vec::<FuncInst>::with_capacity(type_section_count);
        let mut exports = ExportContainer::default();
        let mut memories = vec![];

        if let Some(section) = module.import_section {
            for import in section {
                match import.desc {
                    crate::binary::types::ImportDesc::Func(index) => {   
                        let Some(ref func_types) = module.type_section else {
                            bail!("Not found type section")
                        };
                        let Some(fn_type) = func_types.get(index as usize) else {
                            bail!("Not found func type in section")
                        };

                        fns.push(FuncInst::External(ExternalFuncInst { 
                            fn_type: fn_type.clone(), mod_name: import.mod_name, fn_name: import.field_name, 
                        }))
                    }
                }
            }
        }

        if let Some(section) = module.code_section {
            for (body, i) in section.iter().zip(decded_fns.into_iter()) {
                let Some(ref func_types) = module.type_section else { 
                    bail!("Not found type section");
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

        if let Some(section) = module.export_section {
            exports.lookup = HashMap::from_iter(section.into_iter().map(|x| (x.name.clone(), x)));
        }

        if let Some(section) = module.memory_section {
            for mem in section {
                let sz = mem.initial as usize * PPAGE_SIZE;
                memories.push(MemoryInst { data: vec![0; sz], limit: mem.limit });
            }
        }

        if let Some(section) = module.data_section {
            for data in section {
                let Some(mem) = memories.get_mut(data.page as usize) else {
                    bail!("Memory page is not found");
                };

                let offset = data.offset as usize;
                mem.data[offset..offset + data.bytes.len()].copy_from_slice(&data.bytes);
            }
        }

        Ok(Self { fns, exports, memories })
    }
}