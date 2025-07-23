"""
🔥 Product Knowledge Base - Specific Analysis for Different Product Types
Provides detailed, realistic analysis based on product characteristics
"""
import numpy as np
from typing import Dict, List, Tuple

try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("⚠️ OpenCV not available, using fallback methods")

class ProductAnalyzer:
    """
    Analyzes specific product characteristics for authentic vs fake detection
    """
    
    def __init__(self):
        # Product-specific knowledge base
        self.product_knowledge = {
            'shoes': {
                'authentic_indicators': {
                    'sole_pattern': 'Đế giày có pattern phức tạp, chi tiết sâu',
                    'stitching': 'Đường may đều đặn, không có chỉ thừa',
                    'logo_placement': 'Logo đặt chính xác theo thiết kế gốc',
                    'material_quality': 'Da/vải có texture tự nhiên, không bóng giả',
                    'color_accuracy': 'Màu sắc đúng tone, không bị pha loãng'
                },
                'fake_indicators': {
                    'sole_pattern': 'Pattern đế đơn giản, thiếu chi tiết',
                    'stitching': 'Đường may lỗi, chỉ không đều',
                    'logo_placement': 'Logo lệch vị trí hoặc tỷ lệ sai',
                    'material_quality': 'Vật liệu nhựa giả, bóng không tự nhiên',
                    'color_accuracy': 'Màu sắc sai lệch, thường đậm hoặc nhạt hơn'
                }
            },
            'clothing': {
                'authentic_indicators': {
                    'fabric_texture': 'Vải có cấu trúc sợi tự nhiên, mềm mại',
                    'print_quality': 'In ấn sắc nét, màu không lem',
                    'seam_quality': 'Đường may chuyên nghiệp, overlock đều',
                    'tag_placement': 'Tag đặt đúng vị trí, chữ rõ ràng',
                    'sizing': 'Size chuẩn theo bảng size chính hãng'
                },
                'fake_indicators': {
                    'fabric_texture': 'Vải cứng, không co giãn tự nhiên',
                    'print_quality': 'In mờ, màu lem lỗi',
                    'seam_quality': 'Đường may thô, không overlock',
                    'tag_placement': 'Tag sai vị trí, font chữ lỗi',
                    'sizing': 'Size không chuẩn, thường nhỏ hơn'
                }
            },
            'accessories': {
                'authentic_indicators': {
                    'hardware_quality': 'Kim loại chất lượng, không gỉ sét',
                    'engaving_sharpness': 'Khắc chữ sắc nét, sâu',
                    'surface_finish': 'Bề mặt hoàn thiện mịn màng',
                    'weight_feel': 'Trọng lượng phù hợp với vật liệu',
                    'packaging': 'Hộp đựng chính hãng, chất lượng cao'
                },
                'fake_indicators': {
                    'hardware_quality': 'Kim loại rẻ tiền, dễ gỉ',
                    'engaving_sharpness': 'Khắc chữ nông, không sắc nét',
                    'surface_finish': 'Bề mặt thô ráp, có vết gia công',
                    'weight_feel': 'Quá nhẹ hoặc quá nặng bất thường',
                    'packaging': 'Hộp kém chất lượng hoặc không có'
                }
            }
        }
        
    def detect_product_type(self, image: np.ndarray, analysis: Dict) -> str:
        """
        Detect product type based on image characteristics
        """
        # Simple heuristic based on shape and features
        h, w = image.shape[:2]
        aspect_ratio = w / h
        
        if CV2_AVAILABLE:
            try:
                # Analyze edges to detect product shape
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                edges = cv2.Canny(gray, 50, 150)
                contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    area_ratio = cv2.contourArea(largest_contour) / (h * w)
                    
                    # Heuristic classification
                    if aspect_ratio > 1.5 and area_ratio > 0.3:
                        return 'shoes'
                    elif aspect_ratio < 0.8 and area_ratio > 0.4:
                        return 'clothing'
                    elif area_ratio < 0.3:
                        return 'accessories'
            except Exception as e:
                print(f"⚠️ CV2 processing failed: {e}")
        
        # Fallback method without CV2
        if aspect_ratio > 1.3:
            return 'shoes'
        elif aspect_ratio < 0.9:
            return 'clothing'
        else:
            return 'accessories'
        
    def analyze_product_specific_features(self, image: np.ndarray, analysis: Dict, 
                                        product_type: str, is_fake: bool) -> Dict:
        """
        Analyze product-specific features for detailed explanation
        """
        if product_type not in self.product_knowledge:
            product_type = 'general'
            
        specific_analysis = {}
        
        # Get product knowledge
        if product_type != 'general':
            knowledge = self.product_knowledge[product_type]
            indicators = knowledge['fake_indicators'] if is_fake else knowledge['authentic_indicators']
            
            # Analyze based on product type
            if product_type == 'shoes':
                specific_analysis.update(self._analyze_shoe_features(image, analysis, indicators, is_fake))
            elif product_type == 'clothing':
                specific_analysis.update(self._analyze_clothing_features(image, analysis, indicators, is_fake))
            elif product_type == 'accessories':
                specific_analysis.update(self._analyze_accessory_features(image, analysis, indicators, is_fake))
        
        return specific_analysis
        
    def _analyze_shoe_features(self, image: np.ndarray, analysis: Dict, 
                              indicators: Dict, is_fake: bool) -> Dict:
        """Analyze specific shoe features"""
        shoe_analysis = {}
        
        # Sole pattern analysis
        texture_strength = analysis.get('texture_strength', 0.5)
        if texture_strength > 0.7:
            shoe_analysis['sole_quality'] = "Đế giày có texture phức tạp, chi tiết pattern rõ ràng"
        elif texture_strength > 0.4:
            shoe_analysis['sole_quality'] = "Đế giày có texture trung bình, một số pattern có thể thấy"
        else:
            shoe_analysis['sole_quality'] = "Đế giày mịn, ít texture hoặc pattern đơn giản"
            
        # Stitching quality
        stitching_quality = analysis.get('stitching_quality', 0.5)
        if stitching_quality > 0.6:
            shoe_analysis['stitching'] = "Đường may chính xác, đều đặn theo chuẩn thương hiệu"
        elif stitching_quality > 0.3:
            shoe_analysis['stitching'] = "Đường may tương đối tốt, có thể có một số chỗ không hoàn hảo"
        else:
            shoe_analysis['stitching'] = "Đường may thô, không đều hoặc có lỗi rõ ràng"
            
        # Logo placement and quality
        logo_sharpness = analysis.get('logo_sharpness', 0.5)
        alignment_score = analysis.get('alignment_score', 0.5)
        
        if logo_sharpness > 0.7 and alignment_score > 0.7:
            shoe_analysis['branding'] = "Logo rất sắc nét, đặt đúng vị trí theo thiết kế gốc"
        elif logo_sharpness > 0.4 or alignment_score > 0.4:
            shoe_analysis['branding'] = "Logo khá rõ ràng, vị trí gần như chuẩn"
        else:
            shoe_analysis['branding'] = "Logo mờ hoặc đặt sai vị trí, có thể là hàng copy"
            
        return shoe_analysis
        
    def _analyze_clothing_features(self, image: np.ndarray, analysis: Dict,
                                  indicators: Dict, is_fake: bool) -> Dict:
        """Analyze specific clothing features"""
        clothing_analysis = {}
        
        # Fabric texture
        surface_roughness = analysis.get('surface_roughness', 0.5)
        if surface_roughness > 0.8:
            clothing_analysis['fabric'] = "Vải có cấu trúc sợi phức tạp, chất lượng cao"
        elif surface_roughness > 0.4:
            clothing_analysis['fabric'] = "Vải chất lượng trung bình, cấu trúc sợi cơ bản"
        else:
            clothing_analysis['fabric'] = "Vải mịn đều, có thể là synthetic hoặc blend"
            
        # Print quality
        color_bleeding = analysis.get('color_bleeding', 0.5)
        color_vibrancy = analysis.get('color_vibrancy', 0.5)
        
        if color_bleeding < 0.2 and color_vibrancy > 0.6:
            clothing_analysis['print_quality'] = "In ấn sắc nét, màu không lem, chất lượng cao"
        elif color_bleeding < 0.5 and color_vibrancy > 0.4:
            clothing_analysis['print_quality'] = "In ấn tương đối tốt, màu ổn định"
        else:
            clothing_analysis['print_quality'] = "In ấn kém, màu có thể lem hoặc mờ"
            
        return clothing_analysis
        
    def _analyze_accessory_features(self, image: np.ndarray, analysis: Dict,
                                   indicators: Dict, is_fake: bool) -> Dict:
        """Analyze specific accessory features"""
        accessory_analysis = {}
        
        # Hardware quality
        shine_ratio = analysis.get('shine_ratio', 0.5)
        if shine_ratio > 0.6:
            accessory_analysis['hardware'] = "Kim loại có độ bóng cao, chất lượng premium"
        elif shine_ratio > 0.3:
            accessory_analysis['hardware'] = "Kim loại có độ bóng vừa phải, chất lượng tiêu chuẩn"
        else:
            accessory_analysis['hardware'] = "Bề mặt ít bóng, có thể là kim loại rẻ hoặc coating kém"
            
        # Surface finish
        surface_smoothness = analysis.get('surface_smoothness', 0.5)
        if surface_smoothness > 0.8:
            accessory_analysis['finish'] = "Bề mặt hoàn thiện rất mịn, gia công tỉ mỉ"
        elif surface_smoothness > 0.5:
            accessory_analysis['finish'] = "Bề mặt khá mịn, gia công tiêu chuẩn"
        else:
            accessory_analysis['finish'] = "Bề mặt thô ráp, có thể thấy vết gia công"
            
        return accessory_analysis
        
    def generate_product_specific_explanation(self, product_type: str, specific_analysis: Dict,
                                            is_fake: bool, confidence: float) -> str:
        """Generate product-specific explanation"""
        explanation = []
        
        if product_type == 'shoes':
            explanation.append("🔍 **Phân Tích Chuyên Biệt Cho Giày:**")
            
            if 'sole_quality' in specific_analysis:
                explanation.append(f"• **Đế giày**: {specific_analysis['sole_quality']}")
                
            if 'stitching' in specific_analysis:
                explanation.append(f"• **Đường may**: {specific_analysis['stitching']}")
                
            if 'branding' in specific_analysis:
                explanation.append(f"• **Logo/thương hiệu**: {specific_analysis['branding']}")
                
        elif product_type == 'clothing':
            explanation.append("🔍 **Phân Tích Chuyên Biệt Cho Quần Áo:**")
            
            if 'fabric' in specific_analysis:
                explanation.append(f"• **Chất vải**: {specific_analysis['fabric']}")
                
            if 'print_quality' in specific_analysis:
                explanation.append(f"• **Chất lượng in**: {specific_analysis['print_quality']}")
                
        elif product_type == 'accessories':
            explanation.append("🔍 **Phân Tích Chuyên Biệt Cho Phụ Kiện:**")
            
            if 'hardware' in specific_analysis:
                explanation.append(f"• **Kim loại**: {specific_analysis['hardware']}")
                
            if 'finish' in specific_analysis:
                explanation.append(f"• **Hoàn thiện bề mặt**: {specific_analysis['finish']}")
        
        # Add specific authenticity assessment
        if is_fake:
            if confidence > 0.8:
                explanation.append("\n💡 **Đánh giá**: Nhiều dấu hiệu cho thấy đây là hàng **KHÔNG CHÍNH HÃNG**")
            else:
                explanation.append("\n💡 **Đánh giá**: Có khả năng cao đây là hàng **KHÔNG CHÍNH HÃNG**")
        else:
            if confidence > 0.8:
                explanation.append("\n💡 **Đánh giá**: Các đặc điểm phù hợp với hàng **CHÍNH HÃNG**")
            else:
                explanation.append("\n💡 **Đánh giá**: Có khả năng cao đây là hàng **CHÍNH HÃNG**")
                
        return "\n".join(explanation)
