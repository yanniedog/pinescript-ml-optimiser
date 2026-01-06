"""
Pine Script Parser
Extracts parameters, signal types, and indicator structure from Pine Script files.
"""

import re
import logging
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from enum import Enum

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class SignalType(Enum):
    DISCRETE = "discrete"           # plotshape buy/sell signals
    THRESHOLD = "threshold"         # threshold crossings
    DIRECTIONAL = "directional"     # oscillator direction
    COMBINED = "combined"           # multiple signal types


class PositionType(Enum):
    LONG_ONLY = "long_only"
    SHORT_ONLY = "short_only"
    BOTH = "both"


@dataclass
class Parameter:
    """Represents an optimizable Pine Script parameter."""
    name: str
    param_type: str  # 'int', 'float', 'bool'
    default: Any
    min_val: Optional[float] = None
    max_val: Optional[float] = None
    step: Optional[float] = None
    title: str = ""
    line_number: int = 0
    original_line: str = ""
    
    def get_search_space(self) -> Tuple[Any, Any, Any]:
        """Return (min, max, step) for optimization."""
        if self.param_type == 'bool':
            return (False, True, None)
        
        min_v = self.min_val if self.min_val is not None else (0 if self.param_type == 'int' else 0.0)
        max_v = self.max_val if self.max_val is not None else (self.default * 5 if self.default else 100)
        
        if self.param_type == 'int':
            step = self.step if self.step else 1
            return (int(min_v), int(max_v), int(step))
        else:
            step = self.step if self.step else (max_v - min_v) / 100
            return (float(min_v), float(max_v), float(step))


@dataclass
class SignalInfo:
    """Information about detected signals in the indicator."""
    signal_type: SignalType
    position_type: PositionType
    buy_conditions: List[str] = field(default_factory=list)
    sell_conditions: List[str] = field(default_factory=list)
    threshold_levels: Dict[str, float] = field(default_factory=dict)
    main_indicator_var: str = ""


@dataclass
class ParseResult:
    """Complete parsing result for a Pine Script file."""
    parameters: List[Parameter]
    signal_info: SignalInfo
    indicator_name: str = ""
    version: int = 6
    is_overlay: bool = False
    raw_content: str = ""
    functions: Dict[str, str] = field(default_factory=dict)
    calculations: List[str] = field(default_factory=list)


class PineParser:
    """Parser for Pine Script indicators."""
    
    # Regex patterns for parameter extraction
    INPUT_INT_PATTERN = re.compile(
        r'(\w+)\s*=\s*input\.int\s*\(\s*'
        r'(-?\d+)\s*'  # default value
        r'(?:,\s*(?:title\s*=\s*)?["\']([^"\']*)["\'])?\s*'  # optional title
        r'(?:,\s*minval\s*=\s*(-?\d+))?\s*'  # optional minval
        r'(?:,\s*maxval\s*=\s*(-?\d+))?\s*'  # optional maxval
        r'(?:,\s*step\s*=\s*(\d+))?\s*'  # optional step
        r'\)',
        re.MULTILINE
    )
    
    INPUT_FLOAT_PATTERN = re.compile(
        r'(\w+)\s*=\s*input\.float\s*\(\s*'
        r'(-?[\d.]+)\s*'  # default value
        r'(?:,\s*(?:title\s*=\s*)?["\']([^"\']*)["\'])?\s*'  # optional title
        r'(?:,\s*minval\s*=\s*(-?[\d.]+))?\s*'  # optional minval
        r'(?:,\s*maxval\s*=\s*(-?[\d.]+))?\s*'  # optional maxval
        r'(?:,\s*step\s*=\s*([\d.]+))?\s*'  # optional step
        r'\)',
        re.MULTILINE
    )
    
    INPUT_BOOL_PATTERN = re.compile(
        r'(\w+)\s*=\s*input\.bool\s*\(\s*'
        r'(true|false)\s*'  # default value
        r'(?:,\s*(?:title\s*=\s*)?["\']([^"\']*)["\'])?\s*'
        r'\)',
        re.MULTILINE | re.IGNORECASE
    )
    
    # Alternative patterns for different input styles
    INPUT_ALT_PATTERN = re.compile(
        r'(\w+)\s*=\s*input\.(int|float|bool)\s*\('
        r'([^)]+)\)',
        re.MULTILINE
    )
    
    PLOTSHAPE_PATTERN = re.compile(
        r'plotshape\s*\(\s*(\w+)[^)]*(?:title\s*=\s*["\']([^"\']*)["\'])?[^)]*\)',
        re.MULTILINE
    )
    
    HLINE_PATTERN = re.compile(
        r'hline\s*\(\s*(-?[\d.]+|\w+)',
        re.MULTILINE
    )
    
    PLOT_PATTERN = re.compile(
        r'plot\s*\(\s*(\w+)',
        re.MULTILINE
    )
    
    INDICATOR_PATTERN = re.compile(
        r'indicator\s*\(\s*["\']([^"\']+)["\']',
        re.MULTILINE
    )
    
    VERSION_PATTERN = re.compile(r'//@version=(\d+)')
    
    def __init__(self):
        self.content = ""
        self.lines = []
    
    def parse_file(self, filepath: str) -> ParseResult:
        """Parse a Pine Script file and extract all relevant information."""
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"Pine Script file not found: {filepath}")
        
        self.content = path.read_text(encoding='utf-8')
        self.lines = self.content.split('\n')
        
        # Extract components
        parameters = self._extract_parameters()
        signal_info = self._detect_signals()
        indicator_name = self._extract_indicator_name()
        version = self._extract_version()
        is_overlay = 'overlay=true' in self.content.lower() or 'overlay = true' in self.content.lower()
        functions = self._extract_functions()
        calculations = self._extract_calculations()
        
        logger.info(f"Parsed {filepath}: {len(parameters)} parameters, signal_type={signal_info.signal_type.value}")
        
        return ParseResult(
            parameters=parameters,
            signal_info=signal_info,
            indicator_name=indicator_name,
            version=version,
            is_overlay=is_overlay,
            raw_content=self.content,
            functions=functions,
            calculations=calculations
        )
    
    def _extract_parameters(self) -> List[Parameter]:
        """Extract all input parameters from the Pine Script."""
        parameters = []
        
        # Process each line to get line numbers
        for line_num, line in enumerate(self.lines, 1):
            # Skip comments
            if line.strip().startswith('//'):
                continue
            
            # Try input.int
            match = re.search(
                r'(\w+)\s*=\s*input\.int\s*\(\s*(-?\d+)',
                line
            )
            if match:
                param = self._parse_input_int(line, line_num, match.group(1))
                if param:
                    parameters.append(param)
                continue
            
            # Try input.float
            match = re.search(
                r'(\w+)\s*=\s*input\.float\s*\(\s*(-?[\d.]+)',
                line
            )
            if match:
                param = self._parse_input_float(line, line_num, match.group(1))
                if param:
                    parameters.append(param)
                continue
            
            # Try input.bool
            match = re.search(
                r'(\w+)\s*=\s*input\.bool\s*\(\s*(true|false)',
                line,
                re.IGNORECASE
            )
            if match:
                param = self._parse_input_bool(line, line_num, match.group(1))
                if param:
                    parameters.append(param)
        
        return parameters
    
    def _parse_input_int(self, line: str, line_num: int, var_name: str) -> Optional[Parameter]:
        """Parse an input.int() declaration."""
        # Extract default value
        default_match = re.search(r'input\.int\s*\(\s*(-?\d+)', line)
        if not default_match:
            return None
        default = int(default_match.group(1))
        
        # Extract optional parameters
        title = self._extract_param(line, 'title', '')
        minval = self._extract_param(line, 'minval', None)
        maxval = self._extract_param(line, 'maxval', None)
        step = self._extract_param(line, 'step', None)
        
        return Parameter(
            name=var_name,
            param_type='int',
            default=default,
            min_val=int(minval) if minval else None,
            max_val=int(maxval) if maxval else None,
            step=int(step) if step else None,
            title=title or var_name,
            line_number=line_num,
            original_line=line.strip()
        )
    
    def _parse_input_float(self, line: str, line_num: int, var_name: str) -> Optional[Parameter]:
        """Parse an input.float() declaration."""
        default_match = re.search(r'input\.float\s*\(\s*(-?[\d.]+)', line)
        if not default_match:
            return None
        default = float(default_match.group(1))
        
        title = self._extract_param(line, 'title', '')
        minval = self._extract_param(line, 'minval', None)
        maxval = self._extract_param(line, 'maxval', None)
        step = self._extract_param(line, 'step', None)
        
        return Parameter(
            name=var_name,
            param_type='float',
            default=default,
            min_val=float(minval) if minval else None,
            max_val=float(maxval) if maxval else None,
            step=float(step) if step else None,
            title=title or var_name,
            line_number=line_num,
            original_line=line.strip()
        )
    
    def _parse_input_bool(self, line: str, line_num: int, var_name: str) -> Optional[Parameter]:
        """Parse an input.bool() declaration."""
        default_match = re.search(r'input\.bool\s*\(\s*(true|false)', line, re.IGNORECASE)
        if not default_match:
            return None
        default = default_match.group(1).lower() == 'true'
        
        title = self._extract_param(line, 'title', '')
        
        return Parameter(
            name=var_name,
            param_type='bool',
            default=default,
            title=title or var_name,
            line_number=line_num,
            original_line=line.strip()
        )
    
    def _extract_param(self, line: str, param_name: str, default: Any) -> Any:
        """Extract a named parameter from an input function call."""
        # Try pattern: param_name = value
        pattern = rf'{param_name}\s*=\s*(["\']?)([^,\)\'"]+)\1'
        match = re.search(pattern, line)
        if match:
            value = match.group(2).strip()
            if value:
                return value
        return default
    
    def _detect_signals(self) -> SignalInfo:
        """Detect the signal type and trading direction from the Pine Script."""
        buy_conditions = []
        sell_conditions = []
        threshold_levels = {}
        main_indicator_var = ""
        
        # Check for plotshape signals (discrete signals)
        has_discrete_signals = False
        for match in self.PLOTSHAPE_PATTERN.finditer(self.content):
            condition = match.group(1)
            title = match.group(2) or ""
            title_lower = title.lower()
            
            if any(kw in title_lower for kw in ['buy', 'bull', 'long', 'up']):
                buy_conditions.append(condition)
                has_discrete_signals = True
            elif any(kw in title_lower for kw in ['sell', 'bear', 'short', 'down']):
                sell_conditions.append(condition)
                has_discrete_signals = True
            elif any(kw in condition.lower() for kw in ['buy', 'bull', 'long']):
                buy_conditions.append(condition)
                has_discrete_signals = True
            elif any(kw in condition.lower() for kw in ['sell', 'bear', 'short']):
                sell_conditions.append(condition)
                has_discrete_signals = True
        
        # Check for threshold levels (hlines)
        for match in self.HLINE_PATTERN.finditer(self.content):
            level_str = match.group(1)
            try:
                level = float(level_str)
                # Look for variable name in context
                context = self.content[max(0, match.start()-100):match.end()+100]
                if 'buy' in context.lower() or 'support' in context.lower():
                    threshold_levels['buy'] = level
                elif 'sell' in context.lower() or 'resistance' in context.lower():
                    threshold_levels['sell'] = level
                elif level == 0:
                    threshold_levels['zero'] = level
            except ValueError:
                # It's a variable reference
                if 'buy' in level_str.lower():
                    threshold_levels['buy_var'] = level_str
                elif 'sell' in level_str.lower():
                    threshold_levels['sell_var'] = level_str
        
        # Find main indicator variable (first plot that's not a constant)
        for match in self.PLOT_PATTERN.finditer(self.content):
            var_name = match.group(1)
            if var_name not in ['0', 'na'] and not var_name.startswith('color'):
                main_indicator_var = var_name
                break
        
        # Determine signal type
        if has_discrete_signals and (buy_conditions or sell_conditions):
            signal_type = SignalType.DISCRETE
        elif threshold_levels:
            signal_type = SignalType.THRESHOLD
        elif main_indicator_var:
            signal_type = SignalType.DIRECTIONAL
        else:
            signal_type = SignalType.DIRECTIONAL  # Default
        
        # Determine position type
        if buy_conditions and sell_conditions:
            position_type = PositionType.BOTH
        elif buy_conditions:
            position_type = PositionType.LONG_ONLY
        elif sell_conditions:
            position_type = PositionType.SHORT_ONLY
        else:
            position_type = PositionType.BOTH  # Default for oscillators
        
        return SignalInfo(
            signal_type=signal_type,
            position_type=position_type,
            buy_conditions=buy_conditions,
            sell_conditions=sell_conditions,
            threshold_levels=threshold_levels,
            main_indicator_var=main_indicator_var
        )
    
    def _extract_indicator_name(self) -> str:
        """Extract the indicator name from the indicator() function."""
        match = self.INDICATOR_PATTERN.search(self.content)
        if match:
            return match.group(1)
        return "Unknown Indicator"
    
    def _extract_version(self) -> int:
        """Extract the Pine Script version."""
        match = self.VERSION_PATTERN.search(self.content)
        if match:
            return int(match.group(1))
        return 5  # Default to v5
    
    def _extract_functions(self) -> Dict[str, str]:
        """Extract user-defined functions from the script."""
        functions = {}
        
        # Pattern for function definitions: name(params) =>
        func_pattern = re.compile(
            r'^(\w+)\s*\(([^)]*)\)\s*=>\s*$',
            re.MULTILINE
        )
        
        for match in func_pattern.finditer(self.content):
            func_name = match.group(1)
            # Get the function body (next lines until empty line or new definition)
            start_pos = match.end()
            end_pos = start_pos
            
            lines = self.content[start_pos:].split('\n')
            body_lines = []
            for line in lines:
                stripped = line.strip()
                if not stripped or (stripped and not stripped.startswith(' ') and not stripped.startswith('\t') and '=' in stripped and '=>' not in stripped):
                    break
                body_lines.append(line)
            
            functions[func_name] = '\n'.join(body_lines)
        
        return functions
    
    def _extract_calculations(self) -> List[str]:
        """Extract main calculation lines (assignments with ta.* functions)."""
        calculations = []
        
        # Look for lines with technical analysis functions
        ta_pattern = re.compile(r'^\s*(\w+)\s*=.*ta\.\w+', re.MULTILINE)
        math_pattern = re.compile(r'^\s*(\w+)\s*=.*math\.\w+', re.MULTILINE)
        
        for match in ta_pattern.finditer(self.content):
            calculations.append(match.group(0).strip())
        
        for match in math_pattern.finditer(self.content):
            line = match.group(0).strip()
            if line not in calculations:
                calculations.append(line)
        
        return calculations


def parse_pine_script(filepath: str) -> ParseResult:
    """Convenience function to parse a Pine Script file."""
    parser = PineParser()
    return parser.parse_file(filepath)


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        result = parse_pine_script(sys.argv[1])
        print(f"\nIndicator: {result.indicator_name}")
        print(f"Version: {result.version}")
        print(f"Signal Type: {result.signal_info.signal_type.value}")
        print(f"Position Type: {result.signal_info.position_type.value}")
        print(f"\nParameters ({len(result.parameters)}):")
        for p in result.parameters:
            bounds = f"[{p.min_val}, {p.max_val}]" if p.min_val is not None else "[auto]"
            print(f"  {p.name}: {p.param_type} = {p.default} {bounds}")
    else:
        print("Usage: python pine_parser.py <pine_script_file>")

