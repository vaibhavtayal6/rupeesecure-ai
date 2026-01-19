"""
Security features information for Indian banknotes
"""

def get_security_features(denomination):
    """
    Get security features available for specific denomination
    """
    base_features = {
        'ashok_pillar': True,
        'gandhi_portrait': True,
        'security_thread': True,
        'serial_numbers': True,
        'watermark': True,
        'latent_image': True,
        'microprinting': True,
        'intaglio_printing': True,
        'see_through_register': True,
        'fluorescent_ink': True,
        'identification_mark': True
    }
    
    # Denomination-specific features
    additional_features = {
        10: {
            'color_shift_ink': False,
            'hologram': False,
            'windowed_security_thread': False
        },
        20: {
            'color_shift_ink': False,
            'hologram': False,
            'windowed_security_thread': True
        },
        50: {
            'color_shift_ink': True,
            'hologram': False,
            'windowed_security_thread': True
        },
        100: {
            'color_shift_ink': True,
            'hologram': True,
            'windowed_security_thread': True
        },
        200: {
            'color_shift_ink': True,
            'hologram': True,
            'windowed_security_thread': True,
            'tactile_markings': True
        },
        500: {
            'color_shift_ink': True,
            'hologram': True,
            'windowed_security_thread': True,
            'tactile_markings': True,
            'emerging_technology': True
        },
        2000: {
            'color_shift_ink': True,
            'hologram': True,
            'windowed_security_thread': True,
            'tactile_markings': True,
            'emerging_technology': True,
            'mahatma_gandhi_series': True
        }
    }
    
    features = base_features.copy()
    features.update(additional_features.get(denomination, {}))
    
    return features

def get_feature_descriptions():
    """
    Get detailed descriptions of security features
    """
    return {
        'ashok_pillar': """
        **Ashoka Pillar Emblem**: Located on the left side of the note, this emblem 
        should be sharp and clear. It represents the national emblem of India and 
        should have fine details visible under magnification.
        """,
        
        'gandhi_portrait': """
        **Mahatma Gandhi Portrait**: The portrait should be sharp with fine lines. 
        When held against light, a perfect register of the watermark should be visible. 
        The portrait also has a latent image that shows the denomination.
        """,
        
        'security_thread': """
        **Security Thread**: A embedded thread that appears as a broken line on the 
        front but as a continuous line when held against light. It contains the words 
        'RBI' and the denomination in microtext.
        """,
        
        'serial_numbers': """
        **Serial Numbers**: Each note has two unique serial numbers - one on the left 
        and one on the right. They should be exactly identical and follow the specific 
        format for each denomination.
        """,
        
        'watermark': """
        **Watermark**: When held against light, the portrait of Mahatma Gandhi and 
        the electrotype watermark showing the denomination should be clearly visible.
        """,
        
        'latent_image': """
        **Latent Image**: On the right side of the note, when tilted, the denomination 
        numeral becomes visible in the vertical band next to the Gandhi portrait.
        """,
        
        'color_shift_ink': """
        **Color-Shifting Ink**: The denomination numeral on the right changes color 
        from green to blue when the note is tilted. This feature is present on ₹50 
        and higher denominations.
        """,
        
        'microprinting': """
        **Microprinting**: The words 'RBI' and the denomination are printed in 
        microletters at various places on the note. These should be sharp and legible 
        under magnification.
        """,
        
        'intaglio_printing': """
        **Intaglio Printing**: The portrait of Gandhi, the Reserve Bank seal, and 
        the denomination numerals have raised print that can be felt by touch.
        """,
        
        'see_through_register': """
        **See-through Register**: Floral designs printed on the front and back align 
        perfectly to form the denomination numeral when held against light.
        """,
        
        'fluorescent_ink': """
        **Fluorescent Ink**: The number panels and certain other areas glow under 
        ultraviolet light.
        """,
        
        'identification_mark': """
        **Identification Mark**: For visually impaired persons, there are raised 
        identification marks in different shapes for different denominations.
        """,
        
        'hologram': """
        **Hologram**: On ₹100 and higher denominations, there is a hologram patch 
        that shows the denomination and the RBI logo from different angles.
        """,
        
        'windowed_security_thread': """
        **Windowed Security Thread**: The thread is partially exposed and partially 
        embedded in the paper, appearing as windows on the front.
        """,
        
        'tactile_markings': """
        **Tactile Markings**: Raised printing that can be felt by touch, helping 
        visually impaired people identify the denomination.
        """
    }

def get_denomination_colors():
    """
    Get color schemes for different denominations
    """
    return {
        10: {"primary": "#6B8E23", "secondary": "#9ACD32", "name": "Chocolate Brown"},
        20: {"primary": "#D2691E", "secondary": "#FF7F50", "name": "Reddish Orange"},
        50: {"primary": "#800080", "secondary": "#BA55D3", "name": "Fluorescent Blue"},
        100: {"primary": "#000080", "secondary": "#4169E1", "name": "Lavender"},
        200: {"primary": "#FF69B4", "secondary": "#FFB6C1", "name": "Bright Yellow"},
        500: {"primary": "#8B4513", "secondary": "#A0522D", "name": "Stone Grey"},
        2000: {"primary": "#2E8B57", "secondary": "#3CB371", "name": "Magenta"}
    }

def get_verification_tips():
    """
    Get manual verification tips for users
    """
    return [
        "Feel the raised printing, especially on Gandhi's portrait and denomination numerals",
        "Check the security thread - it should be embedded, not printed",
        "Look for the watermark by holding the note against light",
        "Tilt the note to see the color-shifting ink (₹50+)",
        "Verify the latent image appears when viewing from an angle",
        "Check that serial numbers on left and right match exactly",
        "Use UV light to see fluorescent patterns",
        "Look for microprinting with a magnifying glass",
        "Ensure the see-through register aligns perfectly",
        "Check the hologram changes image when tilted (₹100+)"
    ]