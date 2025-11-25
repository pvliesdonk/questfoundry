from questfoundry.runtime.core.graph_factory import GraphFactory


def test_hook_harvest_graph_compiles():
    gf = GraphFactory()
    graph = gf.create_loop_graph("hook_harvest", context={})
    assert graph is not None
