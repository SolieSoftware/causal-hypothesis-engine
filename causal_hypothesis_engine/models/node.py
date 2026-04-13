from __future__ import annotations

from enum import Enum
from typing import Annotated, Any, Optional
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field
from pydantic.functional_serializers import PlainSerializer
from pydantic.functional_validators import PlainValidator
from typing_extensions import TypedDict


class NodeType(str, Enum):
    Exposure = "Exposure"
    Outcome = "Outcome"
    Confounder = "Confounder"
    Mediator = "Mediator"
    Collider = "Collider"


class MeasurabilityState(str, Enum):
    Hypothetical = "Hypothetical"
    Identified = "Identified"
    Proxied = "Proxied"
    Validated = "Validated"


class BaseNodeMetadata(TypedDict, total=False):
    """Base typed dict for adapter metadata. All fields optional by default."""


class InsuranceNodeMetadata(BaseNodeMetadata, total=False):
    """Typed metadata for the InsuranceClaimsAdapter."""

    claim_type: str
    proxy_variables: list[str]
    data_source: str
    # Arbitrary extra fields within the insurance domain (e.g. GDACS enrichment).
    domain_metadata: dict[str, Any]


# Pydantic v2 validates TypedDicts strictly and would strip unknown keys when
# the annotated type is a base TypedDict with no declared fields.  We keep the
# TypedDict annotation for static type-checking (mypy/pyright) while telling
# Pydantic to treat the value as a plain dict at runtime so that all keys from
# subclass TypedDicts (e.g. InsuranceNodeMetadata) are preserved unchanged.
_AdapterMetadataField = Annotated[
    BaseNodeMetadata,
    PlainValidator(lambda v: dict(v)),
    PlainSerializer(lambda v: dict(v), return_type=dict),
]


class Node(BaseModel):
    model_config = ConfigDict(use_enum_values=False)

    id: str = Field(default_factory=lambda: str(uuid4()))
    label: str
    description: str = ""
    node_type: NodeType
    measurability_state: MeasurabilityState = MeasurabilityState.Hypothetical
    # Store edge IDs rather than Edge objects to avoid circular references.
    edge_ids: list[str] = Field(default_factory=list)
    # Typed adapter metadata — not a bare dict; uses TypedDict-based mixin.
    adapter_metadata: Optional[_AdapterMetadataField] = None  # type: ignore[valid-type]
