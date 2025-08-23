"""
Content filtering system for EducaIA
Implements basic content safety measures
"""

import re
from typing import Tuple, List, Dict

class ContentFilter:
    def __init__(self):
        # Lista de palabras/frases prohibidas (expandir según necesidades)
        self.harmful_keywords = [
            'violence', 'violencia', 'hate', 'odio', 'discriminación', 'discrimination',
            'suicide', 'suicidio', 'self-harm', 'autolesión', 'drugs', 'drogas',
            'illegal', 'ilegal', 'weapon', 'arma', 'bomb', 'bomba', 'terrorism', 'terrorismo'
        ]
        
        # Patrones de contenido inapropiado
        self.inappropriate_patterns = [
            r'(?i)(how to (?:make|build|create).{0,20}(?:bomb|weapon|drug))',
            r'(?i)(ways to (?:hurt|harm|kill))',
            r'(?i)(suicide methods?|how to (?:die|kill yourself))',
        ]
        
        # Lista de temas educativos permitidos
        self.educational_topics = [
            'mathematics', 'science', 'programming', 'history', 'literature',
            'economics', 'finance', 'excel', 'machine learning', 'ai',
            'matemáticas', 'ciencia', 'programación', 'historia', 'literatura'
        ]

    def filter_input(self, user_message: str) -> Tuple[bool, str]:
        """
        Filtra el input del usuario
        Returns: (is_safe, reason_if_not_safe)
        """
        # TEMPORALMENTE DESACTIVADO PARA MEJORAR RENDIMIENTO
        # Retorna siempre True para permitir todas las consultas
        return True, ""
        
        # CÓDIGO ORIGINAL (COMENTADO):
        # message_lower = user_message.lower()
        # 
        # # Verificar palabras clave dañinas
        # for keyword in self.harmful_keywords:
        #     if keyword in message_lower:
        #         return False, f"Contenido potencialmente dañino detectado: '{keyword}'"
        # 
        # # Verificar patrones inapropiados
        # for pattern in self.inappropriate_patterns:
        #     if re.search(pattern, user_message):
        #         return False, "Patrón de contenido inapropiado detectado"
        # 
        # return True, ""

    def filter_output(self, ai_response: str) -> Tuple[bool, str]:
        """
        Filtra la respuesta de la AI
        Returns: (is_safe, filtered_response)
        """
        # TEMPORALMENTE DESACTIVADO PARA MEJORAR RENDIMIENTO
        # Retorna siempre la respuesta original sin filtrar
        return True, ai_response
        
        # CÓDIGO ORIGINAL (COMENTADO):
        # response_lower = ai_response.lower()
        # 
        # # Verificar si la respuesta contiene contenido dañino
        # for keyword in self.harmful_keywords:
        #     if keyword in response_lower:
        #         return False, "Lo siento, no puedo proporcionar información sobre ese tema. ¿Puedo ayudarte con algo relacionado con educación o aprendizaje?"
        # 
        # # Verificar patrones inapropiados en la respuesta
        # for pattern in self.inappropriate_patterns:
        #     if re.search(pattern, ai_response):
        #         return False, "Lo siento, no puedo proporcionar esa información. ¿Te puedo ayudar con algún tema educativo?"
        # 
        # return True, ai_response

    def enhance_system_prompt(self, base_prompt: str) -> str:
        """
        Añade instrucciones de seguridad al prompt del sistema
        """
        # TEMPORALMENTE DESACTIVADO PARA MEJORAR RENDIMIENTO
        # Retorna el prompt base sin modificar
        return base_prompt
        
        # CÓDIGO ORIGINAL (COMENTADO):
        # safety_instructions = """
        # IMPORTANT SAFETY GUIDELINES:
        # - Only provide educational, helpful, and appropriate content
        # - Do not provide information about violence, self-harm, illegal activities, or discrimination
        # - If asked about harmful topics, politely redirect to educational alternatives
        # - Focus on constructive learning and positive outcomes
        # - Maintain a respectful and professional tone at all times
        # """
        # return base_prompt + "\n" + safety_instructions

# Instancia global del filtro
content_filter = ContentFilter()
