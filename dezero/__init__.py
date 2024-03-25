# =============================================================================
# step23.pyからstep32.pyまではsimple_coreを利用
is_simple_core = True  # False
# =============================================================================

if is_simple_core:
    from dezero.core_simple import Variable
    from dezero.core_simple import Function
    from dezero.core_simple import using_config
    from dezero.core_simple import no_grad
    from dezero.core_simple import as_array
    from dezero.core_simple import as_variable
    from dezero.core_simple import setup_variable

setup_variable()