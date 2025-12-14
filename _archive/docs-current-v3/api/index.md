# API Reference

This section documents the QuestFoundry Python API.

## Runtime

The runtime module provides the execution engine.

```{eval-rst}
.. automodule:: questfoundry.runtime.orchestrator
   :members:
   :undoc-members:
```

### State Management

```{eval-rst}
.. automodule:: questfoundry.runtime.state
   :members:
   :undoc-members:
```

### Stores

```{eval-rst}
.. automodule:: questfoundry.runtime.stores.hot_store
   :members:
   :undoc-members:
```

```{eval-rst}
.. automodule:: questfoundry.runtime.stores.cold_store
   :members:
   :undoc-members:
```

## Compiler

The compiler transforms MyST domain files to Python.

```{eval-rst}
.. automodule:: questfoundry.compiler.compile
   :members:
   :undoc-members:
```

## Generated Models

Auto-generated Pydantic models from domain definitions.

```{eval-rst}
.. automodule:: questfoundry.generated.models.artifacts
   :members:
   :undoc-members:
```

```{eval-rst}
.. automodule:: questfoundry.generated.models.enums
   :members:
   :undoc-members:
```

```{toctree}
:maxdepth: 2
:hidden:

runtime
compiler
models
```
