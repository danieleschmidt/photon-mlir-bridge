"""
Internationalization (i18n) support for Quantum-Inspired Task Scheduler

Provides multi-language support for global deployment with support for:
- English (en) - Default
- Spanish (es)
- French (fr)
- German (de)
- Japanese (ja)
- Chinese Simplified (zh)

Global-first design with automatic language detection, fallback mechanisms,
and compliance with international data protection regulations.
"""

import os
import json
import logging
from typing import Dict, Any, Optional, List, Union
from pathlib import Path
from enum import Enum
import re
from datetime import datetime, timezone
import locale
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class SupportedLanguage(Enum):
    """Supported languages with their codes and display names."""
    ENGLISH = ("en", "English", "English", "ltr")
    SPANISH = ("es", "Español", "Spanish", "ltr")  
    FRENCH = ("fr", "Français", "French", "ltr")
    GERMAN = ("de", "Deutsch", "German", "ltr")
    JAPANESE = ("ja", "日本語", "Japanese", "ltr")
    CHINESE_SIMPLIFIED = ("zh", "中文", "Chinese (Simplified)", "ltr")
    
    def __init__(self, code: str, native_name: str, english_name: str, direction: str):
        self.code = code
        self.native_name = native_name
        self.english_name = english_name
        self.direction = direction  # "ltr" or "rtl"
    
    @classmethod
    def from_code(cls, code: str) -> Optional['SupportedLanguage']:
        """Get language enum from language code."""
        for lang in cls:
            if lang.code == code.lower():
                return lang
        return None
    
    @classmethod
    def get_all_codes(cls) -> List[str]:
        """Get list of all supported language codes."""
        return [lang.code for lang in cls]


@dataclass
class LocalizationContext:
    """Context for localization including regional preferences."""
    language: SupportedLanguage = SupportedLanguage.ENGLISH
    region: Optional[str] = None  # e.g., "US", "GB", "CN"
    timezone: str = "UTC"
    currency: str = "USD"
    date_format: str = "%Y-%m-%d"
    time_format: str = "%H:%M:%S"
    number_format: str = "1,234.56"
    rtl_support: bool = False
    
    @property
    def locale_code(self) -> str:
        """Get full locale code (e.g., 'en_US', 'zh_CN')."""
        if self.region:
            return f"{self.language.code}_{self.region}"
        return self.language.code
    
    @property
    def is_rtl(self) -> bool:
        """Check if language is right-to-left."""
        return self.language.direction == "rtl"


class TranslationManager:
    """Manages translation strings and localization."""
    
    def __init__(self, locale_path: Optional[Path] = None):
        self.locale_path = locale_path or Path(__file__).parent / "locales"
        self.translations: Dict[str, Dict[str, str]] = {}
        self.fallback_language = SupportedLanguage.ENGLISH
        self.current_language = SupportedLanguage.ENGLISH
        self.cache_enabled = True
        
        # Initialize locale directory
        self.locale_path.mkdir(parents=True, exist_ok=True)
        
        # Load translations
        self._load_all_translations()
        
        logger.info(f"Translation manager initialized with {len(self.translations)} languages")
    
    def set_language(self, language: Union[str, SupportedLanguage]):
        """Set the current language."""
        if isinstance(language, str):
            lang = SupportedLanguage.from_code(language)
            if not lang:
                logger.warning(f"Unsupported language code: {language}, falling back to {self.fallback_language.code}")
                lang = self.fallback_language
        else:
            lang = language
        
        self.current_language = lang
        logger.debug(f"Language set to: {lang.code} ({lang.native_name})")
    
    def get_language(self) -> SupportedLanguage:
        """Get current language."""
        return self.current_language
    
    def translate(self, key: str, language: Optional[Union[str, SupportedLanguage]] = None, 
                 default: Optional[str] = None, **kwargs) -> str:
        """
        Translate a key to the specified or current language.
        
        Args:
            key: Translation key (e.g., 'quantum.scheduler.status.running')
            language: Target language (defaults to current)
            default: Default text if translation not found
            **kwargs: Variables for string formatting
            
        Returns:
            Translated string
        """
        # Determine target language
        if language is None:
            target_lang = self.current_language
        elif isinstance(language, str):
            target_lang = SupportedLanguage.from_code(language) or self.fallback_language
        else:
            target_lang = language
        
        # Get translation
        translation = self._get_translation(key, target_lang, default)
        
        # Apply string formatting if variables provided
        if kwargs:
            try:
                translation = translation.format(**kwargs)
            except (KeyError, ValueError) as e:
                logger.warning(f"Translation formatting failed for key '{key}': {e}")
        
        return translation
    
    def _get_translation(self, key: str, language: SupportedLanguage, default: Optional[str]) -> str:
        """Get translation for specific language with fallback."""
        # Try target language
        if language.code in self.translations:
            translation = self._extract_nested_key(self.translations[language.code], key)
            if translation:
                return translation
        
        # Try fallback language
        if language != self.fallback_language and self.fallback_language.code in self.translations:
            translation = self._extract_nested_key(self.translations[self.fallback_language.code], key)
            if translation:
                logger.debug(f"Using fallback translation for key '{key}'")
                return translation
        
        # Use provided default or key itself
        result = default or key
        logger.warning(f"Translation not found for key '{key}' in language '{language.code}', using: {result}")
        return result
    
    def _extract_nested_key(self, translations: Dict[str, Any], key: str) -> Optional[str]:
        """Extract value from nested dictionary using dot notation."""
        keys = key.split('.')
        current = translations
        
        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return None
        
        return str(current) if current is not None else None
    
    def _load_all_translations(self):
        """Load all translation files."""
        for language in SupportedLanguage:
            self._load_language_translations(language)
    
    def _load_language_translations(self, language: SupportedLanguage):
        """Load translations for a specific language."""
        translation_file = self.locale_path / f"{language.code}.json"
        
        if not translation_file.exists():
            logger.info(f"Creating default translation file for {language.code}")
            self._create_default_translations(language)
        
        try:
            with open(translation_file, 'r', encoding='utf-8') as f:
                translations = json.load(f)
                self.translations[language.code] = translations
                logger.debug(f"Loaded {len(translations)} translations for {language.code}")
        except Exception as e:
            logger.error(f"Failed to load translations for {language.code}: {e}")
            if language == self.fallback_language:
                # Ensure fallback language has minimal translations
                self.translations[language.code] = self._get_minimal_translations()
    
    def _create_default_translations(self, language: SupportedLanguage):
        """Create default translation file for a language."""
        translations = self._get_default_translations(language)
        
        translation_file = self.locale_path / f"{language.code}.json"
        try:
            with open(translation_file, 'w', encoding='utf-8') as f:
                json.dump(translations, f, ensure_ascii=False, indent=2)
            logger.info(f"Created default translation file: {translation_file}")
        except Exception as e:
            logger.error(f"Failed to create translation file for {language.code}: {e}")
    
    def _get_default_translations(self, language: SupportedLanguage) -> Dict[str, Any]:
        """Get default translations for a language."""
        # Base English translations
        base_translations = {
            "quantum": {
                "scheduler": {
                    "name": "Quantum-Inspired Task Scheduler",
                    "description": "Advanced task scheduling using quantum-inspired optimization algorithms",
                    "status": {
                        "running": "Running",
                        "stopped": "Stopped", 
                        "error": "Error",
                        "initializing": "Initializing",
                        "optimizing": "Optimizing",
                        "completed": "Completed"
                    },
                    "tasks": {
                        "created": "Task created",
                        "validated": "Task validated",
                        "scheduled": "Task scheduled",
                        "executed": "Task executed",
                        "failed": "Task failed"
                    },
                    "optimization": {
                        "annealing": "Quantum annealing in progress",
                        "converged": "Algorithm converged",
                        "improved": "Solution improved",
                        "stagnated": "Solution stagnated"
                    }
                }
            },
            "api": {
                "errors": {
                    "validation_failed": "Input validation failed",
                    "task_not_found": "Task not found",
                    "scheduling_failed": "Scheduling failed",
                    "internal_error": "Internal server error",
                    "rate_limited": "Rate limit exceeded",
                    "unauthorized": "Unauthorized access"
                },
                "success": {
                    "task_created": "Task created successfully",
                    "schedule_optimized": "Schedule optimized successfully",
                    "cache_cleared": "Cache cleared successfully"
                }
            },
            "validation": {
                "messages": {
                    "required_field": "This field is required",
                    "invalid_format": "Invalid format",
                    "out_of_range": "Value out of range", 
                    "circular_dependency": "Circular dependency detected",
                    "resource_limit": "Resource limit exceeded"
                }
            },
            "ui": {
                "buttons": {
                    "start": "Start",
                    "stop": "Stop",
                    "pause": "Pause",
                    "resume": "Resume",
                    "reset": "Reset",
                    "submit": "Submit",
                    "cancel": "Cancel",
                    "save": "Save",
                    "load": "Load",
                    "export": "Export"
                },
                "navigation": {
                    "dashboard": "Dashboard",
                    "tasks": "Tasks",
                    "schedule": "Schedule",
                    "monitoring": "Monitoring",
                    "settings": "Settings",
                    "help": "Help"
                }
            },
            "time": {
                "formats": {
                    "short_date": "%Y-%m-%d",
                    "long_date": "%Y-%m-%d %H:%M:%S",
                    "time_only": "%H:%M:%S"
                },
                "units": {
                    "seconds": "seconds",
                    "minutes": "minutes", 
                    "hours": "hours",
                    "days": "days"
                }
            }
        }
        
        # Language-specific translations
        if language == SupportedLanguage.SPANISH:
            return {
                "quantum": {
                    "scheduler": {
                        "name": "Planificador de Tareas Inspirado en Cuántica",
                        "description": "Planificación avanzada de tareas usando algoritmos de optimización inspirados en la cuántica",
                        "status": {
                            "running": "Ejecutándose",
                            "stopped": "Detenido",
                            "error": "Error", 
                            "initializing": "Inicializando",
                            "optimizing": "Optimizando",
                            "completed": "Completado"
                        },
                        "tasks": {
                            "created": "Tarea creada",
                            "validated": "Tarea validada",
                            "scheduled": "Tarea programada",
                            "executed": "Tarea ejecutada",
                            "failed": "Tarea falló"
                        }
                    }
                },
                "api": {
                    "errors": {
                        "validation_failed": "La validación de entrada falló",
                        "task_not_found": "Tarea no encontrada",
                        "scheduling_failed": "La programación falló",
                        "internal_error": "Error interno del servidor"
                    }
                }
            }
        
        elif language == SupportedLanguage.FRENCH:
            return {
                "quantum": {
                    "scheduler": {
                        "name": "Planificateur de Tâches Inspiré Quantique",
                        "description": "Planification avancée de tâches utilisant des algorithmes d'optimisation inspirés quantiques",
                        "status": {
                            "running": "En cours d'exécution",
                            "stopped": "Arrêté",
                            "error": "Erreur",
                            "initializing": "Initialisation",
                            "optimizing": "Optimisation",
                            "completed": "Terminé"
                        }
                    }
                }
            }
        
        elif language == SupportedLanguage.GERMAN:
            return {
                "quantum": {
                    "scheduler": {
                        "name": "Quanteninspirierter Aufgabenplaner",
                        "description": "Erweiterte Aufgabenplanung mit quanteninspirierten Optimierungsalgorithmen",
                        "status": {
                            "running": "Läuft",
                            "stopped": "Gestoppt",
                            "error": "Fehler",
                            "initializing": "Initialisierung",
                            "optimizing": "Optimierung",
                            "completed": "Abgeschlossen"
                        }
                    }
                }
            }
        
        elif language == SupportedLanguage.JAPANESE:
            return {
                "quantum": {
                    "scheduler": {
                        "name": "量子インスパイアタスクスケジューラ",
                        "description": "量子インスパイア最適化アルゴリズムを使用した高度なタスクスケジューリング",
                        "status": {
                            "running": "実行中",
                            "stopped": "停止",
                            "error": "エラー",
                            "initializing": "初期化中",
                            "optimizing": "最適化中",
                            "completed": "完了"
                        }
                    }
                }
            }
        
        elif language == SupportedLanguage.CHINESE_SIMPLIFIED:
            return {
                "quantum": {
                    "scheduler": {
                        "name": "量子启发任务调度器",
                        "description": "使用量子启发优化算法的高级任务调度",
                        "status": {
                            "running": "运行中",
                            "stopped": "已停止", 
                            "error": "错误",
                            "initializing": "初始化中",
                            "optimizing": "优化中",
                            "completed": "已完成"
                        }
                    }
                }
            }
        
        # Return base translations for English or unknown languages
        return base_translations
    
    def _get_minimal_translations(self) -> Dict[str, Any]:
        """Get minimal translations to ensure basic functionality."""
        return {
            "quantum": {
                "scheduler": {
                    "name": "Quantum Task Scheduler",
                    "status": {
                        "running": "Running",
                        "error": "Error"
                    }
                }
            }
        }
    
    def get_supported_languages(self) -> List[Dict[str, str]]:
        """Get list of supported languages with metadata."""
        return [
            {
                "code": lang.code,
                "native_name": lang.native_name,
                "english_name": lang.english_name,
                "direction": lang.direction
            }
            for lang in SupportedLanguage
        ]
    
    def get_translation_coverage(self, language: Union[str, SupportedLanguage]) -> Dict[str, Any]:
        """Get translation coverage statistics for a language."""
        if isinstance(language, str):
            lang = SupportedLanguage.from_code(language)
        else:
            lang = language
        
        if not lang or lang.code not in self.translations:
            return {"coverage": 0, "total_keys": 0, "translated_keys": 0}
        
        # Count keys recursively
        def count_keys(obj):
            if isinstance(obj, dict):
                return sum(count_keys(v) for v in obj.values())
            else:
                return 1
        
        # Compare with English (reference)
        english_keys = count_keys(self.translations.get("en", {}))
        target_keys = count_keys(self.translations.get(lang.code, {}))
        
        coverage = (target_keys / english_keys * 100) if english_keys > 0 else 0
        
        return {
            "language": lang.code,
            "coverage": round(coverage, 2),
            "total_keys": english_keys,
            "translated_keys": target_keys
        }


class GlobalizationManager:
    """Manages globalization features including localization, compliance, and regional preferences."""
    
    def __init__(self):
        self.translation_manager = TranslationManager()
        self.current_context = LocalizationContext()
        
        # Regional compliance rules
        self.compliance_rules = {
            "EU": {
                "gdpr": True,
                "cookie_consent": True,
                "data_retention_days": 365,
                "right_to_deletion": True,
                "privacy_by_design": True
            },
            "US": {
                "ccpa": True,
                "coppa": True,
                "data_retention_days": 730,
                "privacy_notice": True
            },
            "CN": {
                "pipl": True,  # Personal Information Protection Law
                "data_localization": True,
                "local_storage_required": True
            },
            "CA": {
                "pipeda": True,  # Personal Information Protection and Electronic Documents Act
                "provincial_laws": True
            }
        }
        
    def set_localization_context(self, context: LocalizationContext):
        """Set the current localization context."""
        self.current_context = context
        self.translation_manager.set_language(context.language)
        logger.info(f"Localization context set: {context.locale_code}")
    
    def detect_user_preferences(self, accept_language: Optional[str] = None,
                              user_agent: Optional[str] = None,
                              ip_address: Optional[str] = None) -> LocalizationContext:
        """
        Detect user preferences from HTTP headers and other signals.
        
        Args:
            accept_language: HTTP Accept-Language header
            user_agent: HTTP User-Agent header
            ip_address: Client IP address for geo-location
            
        Returns:
            Detected localization context
        """
        context = LocalizationContext()
        
        # Parse Accept-Language header
        if accept_language:
            preferred_language = self._parse_accept_language(accept_language)
            if preferred_language:
                context.language = preferred_language
        
        # Detect timezone from IP (simplified - in production use proper geo-IP service)
        if ip_address:
            context.timezone = self._detect_timezone_from_ip(ip_address)
        
        # Set system locale as fallback
        try:
            system_locale = locale.getdefaultlocale()[0]
            if system_locale:
                lang_code = system_locale.split('_')[0]
                detected_lang = SupportedLanguage.from_code(lang_code)
                if detected_lang and not accept_language:
                    context.language = detected_lang
        except Exception as e:
            logger.debug(f"Could not detect system locale: {e}")
        
        logger.debug(f"Detected user preferences: {context.locale_code}")
        return context
    
    def _parse_accept_language(self, accept_language: str) -> Optional[SupportedLanguage]:
        """Parse Accept-Language header to determine preferred language."""
        # Parse "en-US,en;q=0.9,es;q=0.8" format
        languages = []
        
        for item in accept_language.split(','):
            item = item.strip()
            if ';q=' in item:
                lang, quality = item.split(';q=')
                try:
                    quality_val = float(quality)
                except ValueError:
                    quality_val = 1.0
            else:
                lang = item
                quality_val = 1.0
            
            # Extract language code (ignore region for now)
            lang_code = lang.split('-')[0].lower()
            languages.append((lang_code, quality_val))
        
        # Sort by quality score
        languages.sort(key=lambda x: x[1], reverse=True)
        
        # Find first supported language
        for lang_code, _ in languages:
            supported_lang = SupportedLanguage.from_code(lang_code)
            if supported_lang:
                return supported_lang
        
        return None
    
    def _detect_timezone_from_ip(self, ip_address: str) -> str:
        """Detect timezone from IP address (simplified implementation)."""
        # In production, use a proper geo-IP service
        # This is a simplified mapping for demo purposes
        region_timezones = {
            "127.0.0.1": "UTC",
            "localhost": "UTC"
        }
        
        return region_timezones.get(ip_address, "UTC")
    
    def format_datetime(self, dt: datetime, format_type: str = "long") -> str:
        """Format datetime according to current locale preferences."""
        if format_type == "short":
            format_str = self.current_context.date_format
        elif format_type == "time":
            format_str = self.current_context.time_format
        else:  # long
            format_str = f"{self.current_context.date_format} {self.current_context.time_format}"
        
        # Convert to user timezone if needed
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        
        return dt.strftime(format_str)
    
    def format_number(self, number: Union[int, float], type_: str = "default") -> str:
        """Format number according to current locale."""
        if type_ == "currency":
            return f"{self.current_context.currency} {number:,.2f}"
        elif type_ == "percentage":
            return f"{number:.1%}"
        else:
            return f"{number:,}"
    
    def get_compliance_requirements(self, region: str) -> Dict[str, Any]:
        """Get compliance requirements for a specific region."""
        return self.compliance_rules.get(region.upper(), {})
    
    def is_gdpr_applicable(self, user_region: Optional[str] = None) -> bool:
        """Check if GDPR compliance is required."""
        if user_region and user_region.upper() in ["EU", "EEA"]:
            return True
        
        # Check current context region
        if hasattr(self.current_context, 'region') and self.current_context.region in ["EU", "EEA"]:
            return True
        
        return False
    
    def translate(self, key: str, **kwargs) -> str:
        """Convenience method to translate using current context."""
        return self.translation_manager.translate(
            key, 
            language=self.current_context.language,
            **kwargs
        )
    
    def get_localized_config(self) -> Dict[str, Any]:
        """Get configuration adapted for current locale."""
        return {
            "language": self.current_context.language.code,
            "locale": self.current_context.locale_code,
            "timezone": self.current_context.timezone,
            "currency": self.current_context.currency,
            "date_format": self.current_context.date_format,
            "time_format": self.current_context.time_format,
            "rtl": self.current_context.is_rtl,
            "supported_languages": self.translation_manager.get_supported_languages()
        }


# Alias for backward compatibility
I18nManager = GlobalizationManager

# Global instance for easy access
_globalization_manager: Optional[GlobalizationManager] = None


def get_globalization_manager() -> GlobalizationManager:
    """Get global instance of GlobalizationManager."""
    global _globalization_manager
    if _globalization_manager is None:
        _globalization_manager = GlobalizationManager()
    return _globalization_manager


def t(key: str, **kwargs) -> str:
    """Convenience function for translation (alias for translate)."""
    return get_globalization_manager().translate(key, **kwargs)


def setup_i18n(language: Union[str, SupportedLanguage] = SupportedLanguage.ENGLISH,
               region: Optional[str] = None,
               timezone: str = "UTC") -> GlobalizationManager:
    """Setup internationalization with specified preferences."""
    manager = get_globalization_manager()
    
    if isinstance(language, str):
        language = SupportedLanguage.from_code(language) or SupportedLanguage.ENGLISH
    
    context = LocalizationContext(
        language=language,
        region=region, 
        timezone=timezone
    )
    
    manager.set_localization_context(context)
    logger.info(f"i18n setup complete: {language.code} ({language.native_name})")
    
    return manager