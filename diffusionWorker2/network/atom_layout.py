# Copyright 2024 DeepMind Technologies Limited
#
# AlphaFold 3 source code is licensed under CC BY-NC-SA 4.0. To view a copy of
# this license, visit https://creativecommons.org/licenses/by-nc-sa/4.0/
#
# To request access to the AlphaFold 3 model parameters, follow the process set
# out at https://github.com/google-deepmind/alphafold3. You may only use these
# if received directly from Google. Use is subject to terms of use available at
# https://github.com/google-deepmind/alphafold3/blob/main/WEIGHTS_TERMS_OF_USE.md


import dataclasses
from collections.abc import Sequence

import numpy as np
import torch


def convertV2(
    gather_idxs,gather_mask,
    arr: torch.Tensor,
    *,
    layout_axes: tuple[int, ...] = (0,),
) -> torch.Tensor:
    """Convert an array from one atom layout to another."""
    # Translate negative indices to the corresponding positives.
    layout_axes = tuple(i if i >= 0 else i + arr.ndim for i in layout_axes)
    # print("covert")
    # Ensure that layout_axes are continuous.
    layout_axes_begin = layout_axes[0]
    layout_axes_end = layout_axes[-1] + 1

    if layout_axes != tuple(range(layout_axes_begin, layout_axes_end)):
        raise ValueError(f'layout_axes must be continuous. Got {layout_axes}.')
    # layout_shape = arr.shape[layout_axes_begin:layout_axes_end]
    # Compute the shape of the input array with flattened layout.
    batch_shape = arr.shape[:layout_axes_begin]
    features_shape = arr.shape[layout_axes_end:]
    prod_dim = 1
    for dim in arr.shape[layout_axes_begin:layout_axes_end]:
        prod_dim *= dim
    # arr_flattened_shape = batch_shape + (torch.prod(torch.tensor(layout_shape)).item(),) + features_shape
    arr_flattened_shape = batch_shape + (prod_dim,) + features_shape
    # Flatten input array and perform the gather.
    arr_flattened = arr.reshape(arr_flattened_shape)
    if layout_axes_begin == 0:
        out_arr = arr_flattened[gather_idxs, ...]
    elif layout_axes_begin == 1:
        out_arr = arr_flattened[:, gather_idxs, ...]
    elif layout_axes_begin == 2:
        out_arr = arr_flattened[:, :, gather_idxs, ...]
    elif layout_axes_begin == 3:
        out_arr = arr_flattened[:, :, :, gather_idxs, ...]
    elif layout_axes_begin == 4:
        out_arr = arr_flattened[:, :, :, :,gather_idxs, ...]
    else:
        raise ValueError(
            'Only 4 batch axes supported. If you need more, the code '
            'is easy to extend.'
        )
    # Broadcast the mask and apply it.

    broadcasted_mask_shape = (
        (1,) * len(batch_shape)
        + gather_mask.shape
        + (1,) * len(features_shape)
    )
    out_arr *= gather_mask.reshape(broadcasted_mask_shape)
    return out_arr


@dataclasses.dataclass(frozen=True)
class AtomLayout:
    """Atom layout in a fixed shape (usually 1-dim or 2-dim).

    Examples for atom layouts are atom37, atom14, and similar.
    All members are np.ndarrays with the same shape, e.g.
    - [num_atoms]
    - [num_residues, max_atoms_per_residue]
    - [num_fragments, max_fragments_per_residue]
    All string arrays should have dtype=object to avoid pitfalls with Numpy's
    fixed-size strings

    Attributes:
      atom_name: np.ndarray of str: atom names (e.g. 'CA', 'NE2'), padding
        elements have an empty string (''), None or any other value, that maps to
        False for .astype(bool). mmCIF field: _atom_site.label_atom_id.
      res_id: np.ndarray of int: residue index (usually starting from 1) padding
        elements can have an arbitrary value. mmCIF field:
        _atom_site.label_seq_id.
      chain_id: np.ndarray of str: chain names (e.g. 'A', 'B') padding elements
        can have an arbitrary value. mmCIF field: _atom_site.label_seq_id.
      atom_element: np.ndarray of str: atom elements (e.g. 'C', 'N', 'O'), padding
        elements have an empty string (''), None or any other value, that maps to
        False for .astype(bool). mmCIF field: _atom_site.type_symbol.
      res_name: np.ndarray of str: residue names (e.g. 'ARG', 'TRP') padding
        elements can have an arbitrary value. mmCIF field:
        _atom_site.label_comp_id.
      chain_type: np.ndarray of str: chain types (e.g. 'polypeptide(L)'). padding
        elements can have an arbitrary value. mmCIF field: _entity_poly.type OR
        _entity.type (for non-polymers).
      shape: shape of the layout (just returns atom_name.shape)
    """

    atom_name: torch.Tensor
    res_id: torch.Tensor
    chain_id: torch.Tensor
    atom_element: torch.Tensor | None = None
    res_name: torch.Tensor | None = None
    chain_type: torch.Tensor | None = None

    def __post_init__(self):
        """Assert all arrays have the same shape."""
        attribute_names = (
            'atom_name',
            'atom_element',
            'res_name',
            'res_id',
            'chain_id',
            'chain_type',
        )
        _assert_all_arrays_have_same_shape(
            obj=self,
            expected_shape=self.atom_name.shape,
            attribute_names=attribute_names,
        )

    def __getitem__(self, key) -> 'AtomLayout':
        return AtomLayout(
            atom_name=self.atom_name[key],
            res_id=self.res_id[key],
            chain_id=self.chain_id[key],
            atom_element=(
                self.atom_element[key] if self.atom_element is not None else None
            ),
            res_name=(self.res_name[key]
                      if self.res_name is not None else None),
            chain_type=(
                self.chain_type[key] if self.chain_type is not None else None
            ),
        )

    def __eq__(self, other: 'AtomLayout') -> bool:
        if not torch.equal(self.atom_name, other.atom_name):
            return False

        mask = self.atom_name.to(dtype=torch.bool)
        # Check essential fields.
        for field in ('res_id', 'chain_id'):
            my_arr = getattr(self, field)
            other_arr = getattr(other, field)
            if not torch.equal(my_arr[mask], other_arr[mask]):
                return False

        # Check optional fields.
        for field in ('atom_element', 'res_name', 'chain_type'):
            my_arr = getattr(self, field)
            other_arr = getattr(other, field)
            if (
                my_arr is not None
                and other_arr is not None
                and not torch.equal(my_arr[mask], other_arr[mask])
            ):
                return False

        return True

    def copy_and_pad_to(self, shape: tuple[int, ...]) -> 'AtomLayout':
        """Copies and pads the layout to the requested shape.

        Args:
          shape: new shape for the atom layout

        Returns:
          a copy of the atom layout padded to the requested shape

        Raises:
          ValueError: incompatible shapes.
        """
        if len(shape) != len(self.atom_name.shape):
            raise ValueError(
                f'Incompatible shape {shape}. Current layout has shape {self.shape}.'
            )
        if any(new < old for old, new in zip(self.atom_name.shape, shape)):
            raise ValueError(
                "Can't pad to a smaller shape. Current layout has shape "
                f'{self.shape} and you requested shape {shape}.'
            )
        pad_width = [
            (0, new - old) for old, new in zip(self.atom_name.shape, shape)
        ]
        pad_val = np.array('', dtype=object)
        return AtomLayout(
            atom_name=np.pad(self.atom_name, pad_width,
                             constant_values=pad_val),
            res_id=np.pad(self.res_id, pad_width, constant_values=0),
            chain_id=np.pad(self.chain_id, pad_width, constant_values=pad_val),
            atom_element=(
                np.pad(self.atom_element, pad_width, constant_values=pad_val)
                if self.atom_element is not None
                else None
            ),
            res_name=(
                np.pad(self.res_name, pad_width, constant_values=pad_val)
                if self.res_name is not None
                else None
            ),
            chain_type=(
                np.pad(self.chain_type, pad_width, constant_values=pad_val)
                if self.chain_type is not None
                else None
            ),
        )

    def to_array(self) -> np.ndarray:
        """Stacks the fields to a numpy array with shape (6, <layout_shape>).

        Creates a pure numpy array of type `object` by stacking the 6 fields of the
        AtomLayout, i.e. (atom_name, atom_element, res_name, res_id, chain_id,
        chain_type). This method together with from_array() provides an easy way to
        apply pure numpy methods like np.concatenate() to `AtomLayout`s.

        Returns:
          np.ndarray of object with shape (6, <layout_shape>), e.g.
          array([['N', 'CA', 'C', ..., 'CB', 'CG', 'CD'],
           ['N', 'C', 'C', ..., 'C', 'C', 'C'],
           ['LEU', 'LEU', 'LEU', ..., 'PRO', 'PRO', 'PRO'],
           [1, 1, 1, ..., 403, 403, 403],
           ['A', 'A', 'A', ..., 'D', 'D', 'D'],
           ['polypeptide(L)', 'polypeptide(L)', ..., 'polypeptide(L)']],
          dtype=object)
        """
        if (
            self.atom_element is None
            or self.res_name is None
            or self.chain_type is None
        ):
            raise ValueError('All optional fields need to be present.')

        return np.stack(dataclasses.astuple(self), axis=0)

    @classmethod
    def from_array(cls, arr: np.ndarray) -> 'AtomLayout':
        """Creates an AtomLayout object from a numpy array with shape (6, ...).

        see also to_array()
        Args:
          arr: np.ndarray of object with shape (6, <layout_shape>)

        Returns:
          AtomLayout object with shape (<layout_shape>)
        """
        if arr.shape[0] != 6:
            raise ValueError(
                'Given array must have shape (6, ...) to match the 6 fields of '
                'AtomLayout (atom_name, atom_element, res_name, res_id, chain_id, '
                f'chain_type). Your array has {arr.shape=}'
            )
        return cls(*arr)

    @property
    def shape(self) -> tuple[int, ...]:
        return self.atom_name.shape

# ref:
# @dataclasses.dataclass(frozen=True)
# class Residues:
#   """List of residues with meta data.

#   Attributes:
#     res_name: np.ndarray of str [num_res], e.g. 'ARG', 'TRP'
#     res_id: np.ndarray of int [num_res]
#     chain_id: np.ndarray of str [num_res], e.g. 'A', 'B'
#     chain_type: np.ndarray of str [num_res], e.g. 'polypeptide(L)'
#     is_start_terminus: np.ndarray of bool [num_res]
#     is_end_terminus: np.ndarray of bool [num_res]
#     deprotonation: (optional) np.ndarray of set() [num_res], e.g. {'HD1', 'HE2'}
#     smiles_string: (optional) np.ndarray of str [num_res], e.g. 'Cc1ccccc1'
#     shape: shape of the layout (just returns res_name.shape)
#   """

#   res_name: np.ndarray
#   res_id: np.ndarray
#   chain_id: np.ndarray
#   chain_type: np.ndarray
#   is_start_terminus: np.ndarray
#   is_end_terminus: np.ndarray
#   deprotonation: np.ndarray | None = None
#   smiles_string: np.ndarray | None = None

#   def __post_init__(self):
#     """Assert all arrays are 1D have the same shape."""
#     attribute_names = (
#         'res_name',
#         'res_id',
#         'chain_id',
#         'chain_type',
#         'is_start_terminus',
#         'is_end_terminus',
#         'deprotonation',
#         'smiles_string',
#     )
#     _assert_all_arrays_have_same_shape(
#         obj=self,
#         expected_shape=(self.res_name.shape[0],),
#         attribute_names=attribute_names,
#     )

#   def __getitem__(self, key: NumpyIndex) -> 'Residues':
#     return Residues(
#         res_name=self.res_name[key],
#         res_id=self.res_id[key],
#         chain_id=self.chain_id[key],
#         chain_type=self.chain_type[key],
#         is_start_terminus=self.is_start_terminus[key],
#         is_end_terminus=self.is_end_terminus[key],
#         deprotonation=(
#             self.deprotonation[key] if self.deprotonation is not None else None
#         ),
#         smiles_string=(
#             self.smiles_string[key] if self.smiles_string is not None else None
#         ),
#     )

#   def __eq__(self, other: 'Residues') -> bool:
#     return all(
#         np.array_equal(getattr(self, field.name), getattr(other, field.name))
#         for field in dataclasses.fields(self)
#     )

#   @property
#   def shape(self) -> tuple[int, ...]:
#     return self.res_name.shape

@dataclasses.dataclass(frozen=True)
class Residues:
    """List of residues with meta data.

    Attributes:
      res_name: np.ndarray of str [num_res], e.g. 'ARG', 'TRP'
      res_id: np.ndarray of int [num_res]
      chain_id: np.ndarray of str [num_res], e.g. 'A', 'B'
      chain_type: np.ndarray of str [num_res], e.g. 'polypeptide(L)'
      is_start_terminus: np.ndarray of bool [num_res]
      is_end_terminus: np.ndarray of bool [num_res]
      deprotonation: (optional) np.ndarray of set() [num_res], e.g. {'HD1', 'HE2'}
      smiles_string: (optional) np.ndarray of str [num_res], e.g. 'Cc1ccccc1'
      shape: shape of the layout (just returns res_name.shape)
    """

    res_name: torch.Tensor
    res_id: torch.Tensor
    chain_id: torch.Tensor
    chain_type: torch.Tensor
    is_start_terminus: torch.Tensor
    is_end_terminus: torch.Tensor
    deprotonation: torch.Tensor | None = None
    smiles_string: torch.Tensor | None = None

    def __post_init__(self):
        """Assert all arrays are 1D have the same shape."""
        attribute_names = (
            'res_name',
            'res_id',
            'chain_id',
            'chain_type',
            'is_start_terminus',
            'is_end_terminus',
            'deprotonation',
            'smiles_string',
        )
        _assert_all_arrays_have_same_shape(
            obj=self,
            expected_shape=(self.res_name.shape[0],),
            attribute_names=attribute_names,
        )

    def __getitem__(self, key) -> 'Residues':
        return Residues(
            res_name=self.res_name[key],
            res_id=self.res_id[key],
            chain_id=self.chain_id[key],
            chain_type=self.chain_type[key],
            is_start_terminus=self.is_start_terminus[key],
            is_end_terminus=self.is_end_terminus[key],
            deprotonation=(
                self.deprotonation[key] if self.deprotonation is not None else None
            ),
            smiles_string=(
                self.smiles_string[key] if self.smiles_string is not None else None
            ),
        )

    def __eq__(self, other: 'Residues') -> bool:
        return all(
            torch.equal(getattr(self, field.name),
                        getattr(other, field.name))
            for field in dataclasses.fields(self)
        )

    @property
    def shape(self) -> tuple[int, ...]:
        return self.res_name.shape

# ref:
# def _assert_all_arrays_have_same_shape(
#     *,
#     obj: AtomLayout | Residues | GatherInfo,
#     expected_shape: tuple[int, ...],
#     attribute_names: Sequence[str],
# ) -> None:
#   """Checks that given attributes of the object have the expected shape."""
#   attribute_shapes_description = []
#   all_shapes_are_valid = True

#   for attribute_name in attribute_names:
#     attribute = getattr(obj, attribute_name)

#     if attribute is None:
#       attribute_shape = None
#     else:
#       attribute_shape = attribute.shape

#     if attribute_shape is not None and expected_shape != attribute_shape:
#       all_shapes_are_valid = False

#     attribute_shape_name = attribute_name + '.shape'
#     attribute_shapes_description.append(
#         f'{attribute_shape_name:25} = {attribute_shape}'
#     )

#   if not all_shapes_are_valid:
#     raise ValueError(
#         f'All arrays must have the same shape ({expected_shape=}). Got\n'
#         + '\n'.join(attribute_shapes_description)
#     )

def _assert_all_arrays_have_same_shape(
    *,
    obj: AtomLayout | Residues ,
    expected_shape: tuple[int, ...],
    attribute_names: Sequence[str],
) -> None:
    """Checks that given attributes of the object have the expected shape."""
    attribute_shapes_description = []
    all_shapes_are_valid = True

    for attribute_name in attribute_names:
        attribute = getattr(obj, attribute_name)

        if attribute is None:
            attribute_shape = None
        else:
            attribute_shape = attribute.shape

        if attribute_shape is not None and expected_shape != attribute_shape:
            all_shapes_are_valid = False

        attribute_shape_name = attribute_name + '.shape'
        attribute_shapes_description.append(
            f'{attribute_shape_name:25} = {attribute_shape}'
        )

    if not all_shapes_are_valid:
        raise ValueError(
            f'All arrays must have the same shape ({expected_shape=}). Got\n'
            + '\n'.join(attribute_shapes_description)
        )
