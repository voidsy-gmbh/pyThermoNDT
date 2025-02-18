from ..units import Unit, Units, is_unit_info
from .base import BaseOps
from .node import DataNode, GroupNode


class AttributeOps(BaseOps):
    def add_attribute(self, path: str, key: str, value: str | int | float | list | tuple | dict | Unit):
        """Adds an attribute to the specified group or dataset in the DataContainer.

        Parameters:
            path (str): The path to the group or dataset.
            key (str): The key of the attribute.
            value (str | int | float | list | tuple | dict | UnitInfo): The value of the attribute.

        Raises:
            KeyError: If the group or dataset does not exist.
            KeyError: If the attribute already exists.
        """
        self.nodes(path, DataNode, GroupNode).add_attribute(key, value)

    def add_attributes(self, path: str, **attributes: str | int | float | list | tuple | dict | Unit):
        """Adds multiple attributes to the specified group or dataset in the DataContainer.

        Parameters:
            path (str): The path to the group or dataset.
            **attributes (Dict[str, str | int | float | list | tuple | dict | UnitInfo]): The attributes to add.

        Raises:
            KeyError: If the group or dataset does not exist.
            KeyError: If any of the attributes already exists.
        """
        self.nodes(path, DataNode, GroupNode).add_attributes(**attributes)

    def add_unit(self, path: str, unit: Unit):
        """Adds the unit information to the specified dataset

        Parameters:
            path (str): The path to the dataset.
            unit (UnitInfo): The unit to add.

        Raises:
            KeyError: If the group or dataset does not exist.
            KeyError: If the unit already exists.
        """
        self.nodes(path, DataNode).add_attribute("Unit", unit)

    def get_attribute(self, path: str, key: str) -> str | int | float | list | tuple | dict | Unit:
        """Get a single attribute from a specified group or dataset in the DataContainer.

        Parameters:
            path (str): The path to the group or dataset.
            key (str): The key of the attribute.

        Returns:
            str | int | float | list | tuple | dict | UnitInfo: The value of the attribute.

        Raises:
            KeyError: If the group or dataset does not exist.
            KeyError: If the attribute does not exist.
        """
        return self.nodes(path, DataNode, GroupNode).get_attribute(key)

    def get_attributes(self, path: str, *keys: str) -> tuple[str | int | float | list | tuple | dict | Unit, ...]:
        """Get multiple attributes from a specified group or dataset in the DataContainer.

        Parameters:
            path (str): The path to the group or dataset.
            *keys (str): Variable number of attribute keys.
                Can be provided as separate arguments or unpacked from a list.

        Returns:
            Tuple[str | int | float | list | tuple | dict | Unit, ...]: The values of the attributes, in the same order.

        Raises:
            KeyError: If the group or dataset does not exist.
            KeyError: If any attribute does not exist.
        """
        return tuple(self.get_attribute(path, key) for key in keys)

    def get_all_attributes(self, path: str) -> dict[str, str | int | float | list | tuple | dict | Unit]:
        """Get all attributes from a specified group or dataset in the DataContainer.

        Parameters:
            path (str): The path to the group or dataset.

        Returns:
            Dict[str, str | int | float | list | tuple | dict | UnitInfo]: A dictionary of all attributes at the path.

        Raises:
            KeyError: If the group or dataset does not exist.
        """
        return dict(self.nodes(path, DataNode, GroupNode).attributes)

    def get_unit(self, path: str) -> Unit:
        """Get the unit information from the specified dataset.

        Parameters:
            path (str): The path to the dataset.

        Returns:
            UnitInfo: The unit information. If no unit information is available, it returns Units.undefined.

        Raises:
            KeyError: If the group or dataset does not exist.
        """
        # Try to get the unit information from the specified dataset
        try:
            unit = self.nodes(path, DataNode).get_attribute("Unit")
        except KeyError:
            unit = Units.undefined

        # Verify that unit is valid and return it ==> otherwise return undefined
        return unit if is_unit_info(unit) else Units.undefined

    def remove_attribute(self, path: str, key: str):
        """Remove an attribute from a specified group or dataset in the DataContainer.

        Parameters:
            path (str): The path to the group or dataset.
            key (str): The key of the attribute.

        Raises:
            KeyError: If the group or dataset does not exist.
            KeyError: If the attribute does not exist.
        """
        self.nodes(path, DataNode, GroupNode).remove_attribute(key)

    def update_attribute(self, path: str, key: str, value: str | int | float | list | tuple | dict | Unit):
        """Update an attribute in a specified group or dataset in the DataContainer.

        Parameters:
            path (str): The path to the group or dataset.
            key (str): The key of the attribute.
            value (str | int | float | list | tuple | dict | UnitInfo): The new value of the attribute.

        Raises:
            KeyError: If the group or dataset does not exist.
            KeyError: If the attribute does not exist.
        """
        self.nodes(path, DataNode, GroupNode).update_attribute(key, value)

    def update_attributes(self, path: str, **attributes: str | int | float | list | tuple | dict | Unit):
        """Update multiple attributes in a specified group or dataset in the DataContainer.

        Parameters:
            path (str): The path to the group or dataset.
            **attributes (Dict[str, str | int | float | list | tuple | dict | UnitInfo]): The new attributes.

        Raises:
            KeyError: If the group or dataset does not exist.
            KeyError: If any of the attributes do not exist.
        """
        self.nodes(path, DataNode, GroupNode).update_attributes(**attributes)

    def update_unit(self, path: str, unit: Unit):
        """Update the unit information of the specified dataset.

        Parameters:
            path (str): The path to the dataset.
            unit (UnitInfo): The new unit information.

        Raises:
            KeyError: If the group or dataset does not exist.
            KeyError: If the unit does not exist.
        """
        self.nodes(path, DataNode).update_attribute("Unit", unit)
