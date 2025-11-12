from questfoundry.utils.resources import list_prompts, list_schemas


def test_list_schemas():
    schemas = list_schemas()
    assert isinstance(schemas, list)
    # Will be populated after spec submodule is initialized


def test_list_prompts():
    prompts = list_prompts()
    assert isinstance(prompts, list)
