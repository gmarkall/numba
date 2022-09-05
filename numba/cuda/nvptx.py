from llvmlite import binding as ll

ll.initialize_all_targets()
ll.initialize_all_asmprinters()


def llvm_to_ptx(irs, options):
    target = ll.Target.from_triple('nvptx64-nvidia-cuda')
    tm = target.create_target_machine()

    mods = [ll.parse_assembly(ir) for ir in irs]
    main_module, *other_modules = mods

    for other in other_modules:
        main_module.link_in(other)

    return tm.emit_assembly(main_module)
