#!/usr/bin/env python3
"""
Internationalization (i18n) Demo for Quantum-Inspired Task Scheduler

Demonstrates multi-language support, localization features, and global-first design
with automatic language detection and compliance with international regulations.
"""

import logging
import time
from datetime import datetime, timezone

import photon_mlir as pm


def setup_logging():
    """Setup logging for the demo."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def demonstrate_basic_translation():
    """Demonstrate basic translation functionality."""
    print("\n" + "="*60)
    print("BASIC TRANSLATION DEMONSTRATION")
    print("="*60)
    
    # Setup English (default)
    manager = pm.setup_i18n("en")
    
    print(f"🌍 Current language: {manager.get_localized_config()['language']}")
    print(f"📝 Scheduler name: {pm.t('quantum.scheduler.name')}")
    print(f"📖 Description: {pm.t('quantum.scheduler.description')}")
    print(f"▶️  Status: {pm.t('quantum.scheduler.status.running')}")
    print(f"✅ Task created: {pm.t('quantum.scheduler.tasks.created')}")
    
    print(f"\n🔧 API Success: {pm.t('api.success.task_created', task_id='demo_001')}")
    print(f"❌ API Error: {pm.t('api.errors.task_not_found', task_id='invalid_123')}")


def demonstrate_multi_language():
    """Demonstrate multiple language support."""
    print("\n" + "="*60)
    print("MULTI-LANGUAGE SUPPORT DEMONSTRATION")
    print("="*60)
    
    languages = [
        ("en", "English"),
        ("es", "Español"),
        ("fr", "Français"),
        ("de", "Deutsch"),
        ("ja", "日本語"),
        ("zh", "中文")
    ]
    
    for lang_code, lang_name in languages:
        print(f"\n🌐 Language: {lang_name} ({lang_code})")
        
        # Setup language
        manager = pm.setup_i18n(lang_code)
        
        # Translate common strings
        scheduler_name = pm.t('quantum.scheduler.name')
        status_running = pm.t('quantum.scheduler.status.running')
        task_created = pm.t('quantum.scheduler.tasks.created')
        
        print(f"   📛 Scheduler: {scheduler_name}")
        print(f"   ⚡ Status: {status_running}")
        print(f"   ✅ Task: {task_created}")
        
        # Show UI elements
        start_btn = pm.t('ui.buttons.start')
        dashboard = pm.t('ui.navigation.dashboard')
        print(f"   🔘 Button: {start_btn}")
        print(f"   📊 Navigation: {dashboard}")


def demonstrate_localization_context():
    """Demonstrate localization context and regional preferences."""
    print("\n" + "="*60)
    print("LOCALIZATION CONTEXT DEMONSTRATION")
    print("="*60)
    
    # Different regional contexts
    contexts = [
        {"lang": "en", "region": "US", "currency": "USD", "timezone": "America/New_York"},
        {"lang": "es", "region": "ES", "currency": "EUR", "timezone": "Europe/Madrid"},
        {"lang": "de", "region": "DE", "currency": "EUR", "timezone": "Europe/Berlin"},
        {"lang": "ja", "region": "JP", "currency": "JPY", "timezone": "Asia/Tokyo"},
        {"lang": "zh", "region": "CN", "currency": "CNY", "timezone": "Asia/Shanghai"}
    ]
    
    current_time = datetime.now(timezone.utc)
    
    for context in contexts:
        print(f"\n🌍 Region: {context['lang'].upper()}_{context['region']}")
        
        # Setup localization
        manager = pm.setup_i18n(context["lang"], context["region"], context["timezone"])
        
        # Get localized configuration
        config = manager.get_localized_config()
        
        print(f"   🗣️  Language: {config['language']}")
        print(f"   🌐 Locale: {config['locale']}")
        print(f"   ⏰ Timezone: {config['timezone']}")
        print(f"   💰 Currency: {config['currency']}")
        
        # Format datetime and numbers
        formatted_time = manager.format_datetime(current_time, "long")
        formatted_number = manager.format_number(12345.67)
        formatted_currency = manager.format_number(99.95, "currency")
        
        print(f"   📅 Time: {formatted_time}")
        print(f"   🔢 Number: {formatted_number}")
        print(f"   💵 Price: {formatted_currency}")
        
        # Show localized scheduler messages
        optimizing = pm.t('quantum.scheduler.status.optimizing')
        print(f"   ⚙️  Status: {optimizing}")


def demonstrate_quantum_scheduler_i18n():
    """Demonstrate i18n in quantum scheduler operations."""
    print("\n" + "="*60)
    print("QUANTUM SCHEDULER I18N INTEGRATION")
    print("="*60)
    
    # Setup Spanish localization
    manager = pm.setup_i18n("es", "ES")
    
    print(f"🔧 Setting up quantum scheduler in {manager.current_context.language.native_name}...")
    
    # Create scheduler with localized messages
    planner = pm.QuantumTaskPlanner()
    
    config = {
        "model_type": "transformer",
        "layers": 6,
        "hidden_size": 512
    }
    
    print(f"📋 {pm.t('api.info.processing')}")
    
    # Generate tasks
    tasks = planner.create_compilation_plan(config)
    
    print(f"✅ {pm.t('api.success.task_created', task_id=f'plan_{len(tasks)}_tasks')}")
    print(f"📊 {pm.t('api.info.progress', current=len(tasks), total=len(tasks), percent=100.0)}")
    
    # Optimize schedule
    print(f"⚡ {pm.t('quantum.scheduler.optimization.annealing')}")
    
    start_time = time.time()
    result = planner.optimize_schedule(tasks)
    optimization_time = time.time() - start_time
    
    print(f"🎯 {pm.t('api.success.schedule_optimized', makespan=result.makespan)}")
    print(f"📈 {pm.t('ui.labels.utilization')}: {result.resource_utilization:.1%}")
    print(f"⏱️  {pm.t('api.info.estimated_time')}: {optimization_time:.2f}s")
    
    # Show optimization details with localized terms
    print(f"\n📊 {pm.t('ui.navigation.metrics')}:")
    print(f"   • {pm.t('ui.labels.makespan')}: {result.makespan:.2f}s")
    print(f"   • {pm.t('ui.labels.utilization')}: {result.resource_utilization:.1%}")
    print(f"   • {pm.t('quantum.scheduler.optimization.converged')}")


def demonstrate_compliance_features():
    """Demonstrate compliance and privacy features."""
    print("\n" + "="*60)
    print("COMPLIANCE AND PRIVACY FEATURES")
    print("="*60)
    
    # European context (GDPR compliance)
    manager = pm.setup_i18n("en", "EU")
    
    print("🇪🇺 European Union Context (GDPR Compliance)")
    
    # Check compliance requirements
    eu_compliance = manager.get_compliance_requirements("EU")
    print(f"   📋 GDPR Required: {eu_compliance.get('gdpr', False)}")
    print(f"   🍪 Cookie Consent: {eu_compliance.get('cookie_consent', False)}")
    print(f"   🗓️  Data Retention: {eu_compliance.get('data_retention_days', 0)} days")
    print(f"   🗑️  Right to Delete: {eu_compliance.get('right_to_deletion', False)}")
    
    # Show GDPR-compliant messages
    if manager.is_gdpr_applicable("EU"):
        print(f"\n🔒 {pm.t('compliance.gdpr.consent_required')}")
        print(f"📄 {pm.t('compliance.gdpr.privacy_policy')}")
        print(f"🍪 {pm.t('compliance.gdpr.cookie_notice')}")
        print(f"📤 {pm.t('compliance.gdpr.data_export')}")
    
    # Security messages
    print(f"\n🛡️  Security Features:")
    print(f"   🔐 {pm.t('compliance.security.data_encrypted')}")
    print(f"   🔒 {pm.t('compliance.security.secure_processing')}")
    print(f"   📝 {pm.t('compliance.security.audit_trail')}")


def demonstrate_language_detection():
    """Demonstrate automatic language detection."""
    print("\n" + "="*60)
    print("AUTOMATIC LANGUAGE DETECTION")
    print("="*60)
    
    manager = pm.GlobalizationManager()
    
    # Simulate different Accept-Language headers
    test_headers = [
        "en-US,en;q=0.9,es;q=0.8",
        "es-ES,es;q=0.9,en;q=0.8",
        "fr-FR,fr;q=0.9,en;q=0.7", 
        "de-DE,de;q=0.9,en;q=0.8",
        "ja-JP,ja;q=0.9,en;q=0.8",
        "zh-CN,zh;q=0.9,en;q=0.7"
    ]
    
    for header in test_headers:
        print(f"\n🌐 Accept-Language: {header}")
        
        # Detect preferences
        context = manager.detect_user_preferences(accept_language=header)
        
        print(f"   🎯 Detected Language: {context.language.code} ({context.language.native_name})")
        print(f"   🏷️  English Name: {context.language.english_name}")
        print(f"   📍 Locale: {context.locale_code}")
        
        # Set detected context and show localized message
        manager.set_localization_context(context)
        scheduler_name = manager.translate('quantum.scheduler.name')
        print(f"   📛 Localized Name: {scheduler_name}")


def demonstrate_translation_coverage():
    """Demonstrate translation coverage analysis."""
    print("\n" + "="*60)
    print("TRANSLATION COVERAGE ANALYSIS")
    print("="*60)
    
    manager = pm.GlobalizationManager()
    
    # Get supported languages
    languages = manager.translation_manager.get_supported_languages()
    
    print("📊 Translation Coverage Report:")
    print(f"{'Language':<15} {'Native Name':<20} {'Coverage':<10} {'Status':<10}")
    print("-" * 65)
    
    for lang in languages:
        coverage = manager.translation_manager.get_translation_coverage(lang['code'])
        
        # Determine status
        if coverage['coverage'] >= 90:
            status = "✅ Complete"
        elif coverage['coverage'] >= 70:
            status = "🟡 Good"
        elif coverage['coverage'] >= 50:
            status = "🟠 Partial"
        else:
            status = "🔴 Basic"
        
        print(f"{lang['english_name']:<15} {lang['native_name']:<20} "
              f"{coverage['coverage']:>6.1f}% {status:<10}")
    
    print(f"\n📈 Translation Statistics:")
    print(f"   • Total Supported Languages: {len(languages)}")
    print(f"   • Fully Translated (>90%): {sum(1 for lang in languages if manager.translation_manager.get_translation_coverage(lang['code'])['coverage'] >= 90)}")
    print(f"   • Partially Translated (50-90%): {sum(1 for lang in languages if 50 <= manager.translation_manager.get_translation_coverage(lang['code'])['coverage'] < 90)}")


def demonstrate_real_world_scenario():
    """Demonstrate a real-world scenario with full i18n integration."""
    print("\n" + "="*60)
    print("REAL-WORLD SCENARIO: GLOBAL DEPLOYMENT")
    print("="*60)
    
    # Simulate users from different regions
    global_users = [
        {"region": "North America", "lang": "en", "locale_region": "US", "timezone": "America/New_York"},
        {"region": "Europe", "lang": "de", "locale_region": "DE", "timezone": "Europe/Berlin"},
        {"region": "Latin America", "lang": "es", "locale_region": "MX", "timezone": "America/Mexico_City"},
        {"region": "Asia-Pacific", "lang": "ja", "locale_region": "JP", "timezone": "Asia/Tokyo"},
        {"region": "China", "lang": "zh", "locale_region": "CN", "timezone": "Asia/Shanghai"}
    ]
    
    for user in global_users:
        print(f"\n🌍 User from {user['region']}")
        
        # Setup user's localization
        manager = pm.setup_i18n(user["lang"], user["locale_region"], user["timezone"])
        
        # Simulate quantum scheduling request
        print(f"   👤 Language: {manager.current_context.language.native_name}")
        print(f"   🏳️  Region: {user['locale_region']}")
        
        # Show localized interface
        dashboard = pm.t('ui.navigation.dashboard')
        start_btn = pm.t('ui.buttons.start')
        task_created = pm.t('quantum.scheduler.tasks.created')
        
        print(f"   🖥️  Dashboard: {dashboard}")
        print(f"   ▶️  Action: {start_btn}")
        print(f"   💬 Message: {task_created}")
        
        # Check regional compliance
        if user["locale_region"] in ["DE", "FR", "IT", "ES"]:  # EU countries
            if manager.is_gdpr_applicable("EU"):
                gdpr_notice = pm.t('compliance.gdpr.consent_required')
                print(f"   ⚖️  Compliance: {gdpr_notice}")
        
        # Show time and currency formatting
        current_time = datetime.now(timezone.utc)
        formatted_time = manager.format_datetime(current_time)
        formatted_price = manager.format_number(199.99, "currency")
        
        print(f"   ⏰ Local Time: {formatted_time}")
        print(f"   💰 Price: {formatted_price}")


def main():
    """Main demo function."""
    setup_logging()
    
    print("🌍 QUANTUM-INSPIRED TASK SCHEDULER - INTERNATIONALIZATION DEMO")
    print("🚀 Global-First Design with Multi-Language Support")
    
    try:
        # Run all demonstrations
        demonstrate_basic_translation()
        demonstrate_multi_language()
        demonstrate_localization_context()
        demonstrate_quantum_scheduler_i18n()
        demonstrate_compliance_features()
        demonstrate_language_detection()
        demonstrate_translation_coverage()
        demonstrate_real_world_scenario()
        
        print("\n" + "="*60)
        print("✅ INTERNATIONALIZATION DEMO COMPLETED SUCCESSFULLY")
        print("="*60)
        print("\n🎉 The Quantum-Inspired Task Scheduler is ready for global deployment!")
        print("🌐 Supports 6 languages with automatic detection and regional compliance")
        print("🛡️  Includes GDPR, CCPA, and other privacy regulation compliance")
        print("⚡ Optimized for multi-cultural teams and international users")
        
    except Exception as e:
        logging.error(f"Demo failed: {e}")
        print(f"\n❌ Demo encountered an error: {e}")
        raise


if __name__ == "__main__":
    main()