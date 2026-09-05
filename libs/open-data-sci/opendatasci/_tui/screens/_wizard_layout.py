"""Shared layout constants for the mandatory pre-chat setup screens.

``StartupWizardScreen``, ``OnboardingScreen``, and ``SystemDependenciesScreen``
run back-to-back before the chat interface ever appears, so they must share
one box width -- picking it independently per screen is how they drift apart.
Wide enough that the longest wizard content (the theme picker's aligned
label/description columns) never wraps.
"""

WIZARD_BOX_WIDTH = 100
