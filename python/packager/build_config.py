"""Build configuration"""

import dataclasses
from typing import Any, Dict, List, Optional


@dataclasses.dataclass
class BuildConfiguration:  # pylint: disable=R0902
    """Configurations use when building libtreelite"""

    # Whether to enable OpenMP
    use_openmp: bool = True
    # Whether to hide C++ symbols
    hide_cxx_symbols: bool = True
    # Whether to use the Treelite library that's installed in the system prefix
    use_system_libtreelite: bool = False

    def _set_config_setting(self, config_settings: Dict[str, Any]) -> None:
        for field_name in config_settings:
            setattr(
                self,
                field_name,
                (config_settings[field_name].lower() in ["true", "1", "on"]),
            )

    def update(self, config_settings: Optional[Dict[str, Any]]) -> None:
        """Parse config_settings from Pip (or other PEP 517 frontend)"""
        if config_settings is not None:
            self._set_config_setting(config_settings)

    def get_cmake_args(self) -> List[str]:
        """Convert build configuration to CMake args"""
        cmake_args = []
        for field_name in [x.name for x in dataclasses.fields(self)]:
            if field_name in ["use_system_libtreelite"]:
                continue
            cmake_option = field_name.upper()
            cmake_value = "ON" if getattr(self, field_name) is True else "OFF"
            cmake_args.append(f"-D{cmake_option}={cmake_value}")
        return cmake_args
