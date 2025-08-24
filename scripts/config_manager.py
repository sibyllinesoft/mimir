#!/usr/bin/env python3
"""
Configuration management CLI tool for Mimir.

Provides utilities for managing, validating, and migrating Mimir configuration.
"""

import argparse
import json
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from repoindex.config import (
    MimirConfig,
    get_config,
    load_config_from_file,
    print_config_summary,
    validate_config,
)
from repoindex.config_migration import migration_tracker


def cmd_validate(args):
    """Validate current configuration."""
    print("Validating Mimir configuration...")
    
    try:
        config = get_config()
        errors = config.validate()
        
        if errors:
            print(f"❌ Configuration validation failed with {len(errors)} issues:")
            for i, error in enumerate(errors, 1):
                print(f"  {i}. {error}")
            return 1
        else:
            print("✅ Configuration validation passed!")
            return 0
            
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        return 1


def cmd_show(args):
    """Show current configuration."""
    try:
        if args.format == "summary":
            print_config_summary()
        elif args.format == "json":
            config = get_config()
            config_dict = config.to_dict()
            
            if args.section:
                if args.section in config_dict:
                    config_dict = {args.section: config_dict[args.section]}
                else:
                    print(f"❌ Section '{args.section}' not found")
                    return 1
            
            print(json.dumps(config_dict, indent=2, default=str))
        elif args.format == "env":
            # Show as environment variables
            config = get_config()
            _print_as_env_vars(config, args.section)
        
        return 0
        
    except Exception as e:
        print(f"❌ Failed to show configuration: {e}")
        return 1


def cmd_export(args):
    """Export configuration to file."""
    try:
        config = get_config()
        output_file = Path(args.output)
        
        if args.format == "json":
            config.save_to_file(output_file)
            print(f"✅ Configuration exported to {output_file}")
        elif args.format == "env":
            _export_as_env_file(config, output_file)
            print(f"✅ Environment variables exported to {output_file}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Failed to export configuration: {e}")
        return 1


def cmd_import(args):
    """Import configuration from file."""
    try:
        config_file = Path(args.file)
        if not config_file.exists():
            print(f"❌ Configuration file not found: {config_file}")
            return 1
        
        config = load_config_from_file(config_file)
        
        if args.validate:
            errors = config.validate()
            if errors:
                print(f"❌ Imported configuration has {len(errors)} validation issues:")
                for error in errors:
                    print(f"  - {error}")
                return 1
        
        print(f"✅ Configuration imported from {config_file}")
        return 0
        
    except Exception as e:
        print(f"❌ Failed to import configuration: {e}")
        return 1


def cmd_diff(args):
    """Compare configurations."""
    try:
        current_config = get_config()
        
        if args.file:
            other_config = MimirConfig.load_from_file(Path(args.file))
            print(f"Comparing current config with {args.file}:")
        else:
            # Compare with default config
            other_config = MimirConfig()
            print("Comparing current config with defaults:")
        
        _diff_configs(current_config, other_config)
        return 0
        
    except Exception as e:
        print(f"❌ Failed to compare configurations: {e}")
        return 1


def cmd_migration_status(args):
    """Show configuration migration status."""
    try:
        migration_tracker.print_migration_status()
        
        # Also show validation status
        print("\nConfiguration Validation:")
        if validate_config():
            print("✅ Current configuration is valid")
        else:
            print("❌ Current configuration has issues (see above)")
        
        return 0
        
    except Exception as e:
        print(f"❌ Failed to show migration status: {e}")
        return 1


def cmd_generate_template(args):
    """Generate configuration template."""
    try:
        config = MimirConfig()  # Default configuration
        
        if args.format == "env":
            _generate_env_template(config, Path(args.output) if args.output else None)
        elif args.format == "json":
            output_file = Path(args.output) if args.output else Path("mimir_config_template.json")
            config.save_to_file(output_file)
            print(f"✅ Configuration template generated: {output_file}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Failed to generate template: {e}")
        return 1


def cmd_check_env(args):
    """Check environment variables."""
    try:
        import os
        
        # List of all environment variables used by Mimir
        env_vars = [
            "MIMIR_LOG_LEVEL", "MIMIR_UI_HOST", "MIMIR_UI_PORT", "MIMIR_TIMEOUT",
            "MIMIR_MAX_WORKERS", "MIMIR_DATA_PATH", "MIMIR_CACHE_PATH",
            "GOOGLE_API_KEY", "GEMINI_API_KEY", "GEMINI_MODEL",
            "MIMIR_ENABLE_METRICS", "MIMIR_SERVICE_NAME",
            "PIPELINE_TIMEOUT_ACQUIRE", "PIPELINE_TIMEOUT_SERENA",
            # Add more as needed
        ]
        
        print("Environment Variables Check:")
        print("=" * 40)
        
        set_vars = []
        unset_vars = []
        
        for var in sorted(env_vars):
            value = os.environ.get(var)
            if value:
                set_vars.append((var, value))
            else:
                unset_vars.append(var)
        
        if set_vars:
            print(f"\nSet variables ({len(set_vars)}):")
            for var, value in set_vars:
                # Mask sensitive values
                if "KEY" in var or "PASSWORD" in var or "SECRET" in var:
                    display_value = value[:8] + "..." if len(value) > 8 else "***"
                else:
                    display_value = value
                print(f"  {var} = {display_value}")
        
        if unset_vars:
            print(f"\nUnset variables ({len(unset_vars)}):")
            for var in unset_vars:
                print(f"  {var}")
        
        return 0
        
    except Exception as e:
        print(f"❌ Failed to check environment: {e}")
        return 1


# Helper functions

def _print_as_env_vars(config: MimirConfig, section: str | None = None):
    """Print configuration as environment variables."""
    config_dict = config.to_dict()
    
    if section:
        if section in config_dict:
            config_dict = {section: config_dict[section]}
        else:
            print(f"❌ Section '{section}' not found")
            return
    
    for section_name, section_data in config_dict.items():
        if section_name.startswith("_"):
            continue
            
        print(f"# {section_name.upper()} CONFIGURATION")
        _print_section_env_vars(section_data, section_name.upper())
        print()


def _print_section_env_vars(data: dict, prefix: str):
    """Print section data as environment variables."""
    for key, value in data.items():
        if isinstance(value, dict):
            _print_section_env_vars(value, f"{prefix}_{key.upper()}")
        else:
            env_var = f"{prefix}_{key.upper()}" if prefix != "MIMIR" else f"MIMIR_{key.upper()}"
            if isinstance(value, list):
                value = ",".join(str(v) for v in value)
            elif isinstance(value, bool):
                value = "true" if value else "false"
            print(f"{env_var}={value}")


def _export_as_env_file(config: MimirConfig, output_file: Path):
    """Export configuration as .env file."""
    config_dict = config.to_dict()
    
    with open(output_file, "w") as f:
        f.write("# Mimir Configuration - Generated Environment Variables\n")
        f.write("# Generated from centralized configuration\n\n")
        
        for section_name, section_data in config_dict.items():
            if section_name.startswith("_"):
                continue
                
            f.write(f"# {section_name.upper()} CONFIGURATION\n")
            _write_section_env_vars(f, section_data, section_name.upper())
            f.write("\n")


def _write_section_env_vars(f, data: dict, prefix: str):
    """Write section data as environment variables to file."""
    for key, value in data.items():
        if isinstance(value, dict):
            _write_section_env_vars(f, value, f"{prefix}_{key.upper()}")
        else:
            env_var = f"{prefix}_{key.upper()}" if prefix != "MIMIR" else f"MIMIR_{key.upper()}"
            if isinstance(value, list):
                value = ",".join(str(v) for v in value)
            elif isinstance(value, bool):
                value = "true" if value else "false"
            f.write(f"{env_var}={value}\n")


def _diff_configs(config1: MimirConfig, config2: MimirConfig):
    """Compare two configurations and show differences."""
    dict1 = config1.to_dict()
    dict2 = config2.to_dict()
    
    differences = []
    _find_dict_differences(dict1, dict2, "", differences)
    
    if differences:
        print("Differences found:")
        for diff in differences:
            print(f"  {diff}")
    else:
        print("✅ Configurations are identical")


def _find_dict_differences(d1: dict, d2: dict, path: str, differences: list):
    """Recursively find differences between dictionaries."""
    all_keys = set(d1.keys()) | set(d2.keys())
    
    for key in sorted(all_keys):
        current_path = f"{path}.{key}" if path else key
        
        if key not in d1:
            differences.append(f"+ {current_path}: {d2[key]} (only in second)")
        elif key not in d2:
            differences.append(f"- {current_path}: {d1[key]} (only in first)")
        elif isinstance(d1[key], dict) and isinstance(d2[key], dict):
            _find_dict_differences(d1[key], d2[key], current_path, differences)
        elif d1[key] != d2[key]:
            differences.append(f"~ {current_path}: {d1[key]} → {d2[key]}")


def _generate_env_template(config: MimirConfig, output_file: Path | None = None):
    """Generate .env template file."""
    if output_file is None:
        output_file = Path(".env.template")
    
    config_dict = config.to_dict()
    
    with open(output_file, "w") as f:
        f.write("# Mimir Configuration Template\n")
        f.write("# Copy this file to .env and customize for your environment\n")
        f.write("# Generated from centralized configuration system\n\n")
        
        for section_name, section_data in config_dict.items():
            if section_name.startswith("_"):
                continue
                
            f.write(f"# ============================================================================\n")
            f.write(f"# {section_name.upper()} CONFIGURATION\n")
            f.write(f"# ============================================================================\n\n")
            
            _write_template_section(f, section_data, section_name.upper())
            f.write("\n")
    
    print(f"✅ Environment template generated: {output_file}")


def _write_template_section(f, data: dict, prefix: str):
    """Write section as commented template."""
    for key, value in data.items():
        if isinstance(value, dict):
            _write_template_section(f, value, f"{prefix}_{key.upper()}")
        else:
            env_var = f"{prefix}_{key.upper()}" if prefix != "MIMIR" else f"MIMIR_{key.upper()}"
            if isinstance(value, list):
                value = ",".join(str(v) for v in value)
            elif isinstance(value, bool):
                value = "true" if value else "false"
            
            # Write as comment with default value
            f.write(f"# {env_var}={value}\n")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Mimir Configuration Management Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Validate command
    validate_parser = subparsers.add_parser("validate", help="Validate current configuration")
    validate_parser.set_defaults(func=cmd_validate)
    
    # Show command
    show_parser = subparsers.add_parser("show", help="Show current configuration")
    show_parser.add_argument("--format", choices=["summary", "json", "env"], default="summary")
    show_parser.add_argument("--section", help="Show specific configuration section")
    show_parser.set_defaults(func=cmd_show)
    
    # Export command
    export_parser = subparsers.add_parser("export", help="Export configuration to file")
    export_parser.add_argument("--output", "-o", required=True, help="Output file path")
    export_parser.add_argument("--format", choices=["json", "env"], default="json")
    export_parser.set_defaults(func=cmd_export)
    
    # Import command
    import_parser = subparsers.add_parser("import", help="Import configuration from file")
    import_parser.add_argument("file", help="Configuration file to import")
    import_parser.add_argument("--validate", action="store_true", help="Validate after import")
    import_parser.set_defaults(func=cmd_import)
    
    # Diff command
    diff_parser = subparsers.add_parser("diff", help="Compare configurations")
    diff_parser.add_argument("--file", help="Compare with configuration file (default: compare with defaults)")
    diff_parser.set_defaults(func=cmd_diff)
    
    # Migration status command
    migration_parser = subparsers.add_parser("migration-status", help="Show migration status")
    migration_parser.set_defaults(func=cmd_migration_status)
    
    # Generate template command
    template_parser = subparsers.add_parser("generate-template", help="Generate configuration template")
    template_parser.add_argument("--output", "-o", help="Output file path")
    template_parser.add_argument("--format", choices=["json", "env"], default="env")
    template_parser.set_defaults(func=cmd_generate_template)
    
    # Check environment command
    env_parser = subparsers.add_parser("check-env", help="Check environment variables")
    env_parser.set_defaults(func=cmd_check_env)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        return args.func(args)
    except KeyboardInterrupt:
        print("\n❌ Operation cancelled")
        return 1
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())