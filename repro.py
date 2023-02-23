
import llvmlite.binding as llvm
import numpy as np

def load_ir(i):
    with open(f'ir_{i}_to_parse.ll', 'r') as f:
        return f.read()

irs = [load_ir(i) for i in range(8)]

llvm.initialize()
llvm.initialize_native_target()
llvm.initialize_native_asmprinter()

lljit = llvm.create_lljit_compiler()

for i, ir in enumerate(irs):
    mod = llvm.parse_assembly(ir)
    print(f"{i}: Adding IR for {mod.name}")
    lljit.add_ir_module(mod)
