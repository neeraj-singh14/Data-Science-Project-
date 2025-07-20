
import json
from typing import List, Dict, Tuple, Optional
import numpy as np
from pymongo import MongoClient

class DSATWhatIfAnalyzer:
    def __init__(self, scoring_maps: Dict):
        """
        Initialize with scoring maps from your JSON data
        
        Args:
            scoring_maps: Dictionary containing 'Math' and 'Reading and Writing' scoring maps
        """
        self.scoring_maps = scoring_maps
        self.subjects = ['Math', 'Reading and Writing']
    
    def get_scaled_score(self, subject: str, raw_score: int, difficulty_level: str) -> int:
        """
        Get scaled score from raw score and difficulty level
        
        Args:
            subject: 'Math' or 'Reading and Writing'
            raw_score: Number of correct answers
            difficulty_level: 'hard' or 'easy'
        
        Returns:
            Scaled score (200-800)
        """
        score_map = self.scoring_maps[subject]['map']
        
        # Find the mapping for this raw score
        for mapping in score_map:
            if mapping['raw'] == raw_score:
                return mapping[difficulty_level]
        
        # If raw score not found, return boundary values
        if raw_score < 0:
            return 200
        elif raw_score >= len(score_map):
            return score_map[-1][difficulty_level]
        
        return 200
    
    def determine_module2_difficulty(self, module1_performance: float, threshold: float = 0.5) -> str:
        """
        Determine Module 2 difficulty based on Module 1 performance
        
        Args:
            module1_performance: Percentage of correct answers in Module 1
            threshold: Threshold for determining hard vs easy Module 2
        
        Returns:
            'hard' or 'easy'
        """
        return 'hard' if module1_performance >= threshold else 'easy'
    
    def calculate_current_score(self, student_responses: List[Dict]) -> Tuple[int, str]:
        """
        Calculate current score and Module 2 difficulty level
        
        Args:
            student_responses: List of response dictionaries containing:
                - question_id: str
                - subject: str ('Math' or 'Reading and Writing')
                - module: int (1 or 2)
                - correct: bool
                - difficulty: str (question difficulty level)
        
        Returns:
            Tuple of (current_scaled_score, module2_difficulty)
        """
        # Separate responses by subject and module
        subject_data = {}
        for subject in self.subjects:
            subject_data[subject] = {
                'module1': [],
                'module2': [],
                'total_correct': 0
            }
        
        # Process responses
        for response in student_responses:
            subject = response['subject']
            module = response['module']
            correct = response['correct']
            
            if module == 1:
                subject_data[subject]['module1'].append(response)
            else:
                subject_data[subject]['module2'].append(response)
            
            if correct:
                subject_data[subject]['total_correct'] += 1
        
        # Calculate scores for each subject
        total_score = 0
        module2_difficulties = {}
        
        for subject in self.subjects:
            # Calculate Module 1 performance
            module1_correct = sum(1 for r in subject_data[subject]['module1'] if r['correct'])
            module1_total = len(subject_data[subject]['module1'])
            module1_performance = module1_correct / module1_total if module1_total > 0 else 0
            
            # Determine Module 2 difficulty
            module2_difficulty = self.determine_module2_difficulty(module1_performance)
            module2_difficulties[subject] = module2_difficulty
            
            # Calculate subject score
            raw_score = subject_data[subject]['total_correct']
            scaled_score = self.get_scaled_score(subject, raw_score, module2_difficulty)
            total_score += scaled_score
        
        return total_score, module2_difficulties
    
    def calculate_impact_score(self, student_responses: List[Dict], 
                             question_to_change: str, 
                             current_total_score: int,
                             current_module2_difficulties: Dict) -> float:
        """
        Calculate the impact of changing a specific question from incorrect to correct
        
        Args:
            student_responses: Original student responses
            question_to_change: ID of question to flip from incorrect to correct
            current_total_score: Current total score
            current_module2_difficulties: Current Module 2 difficulties
        
        Returns:
            Score improvement (can be negative due to adaptive effects)
        """
        # Create modified responses
        modified_responses = []
        target_question = None
        
        for response in student_responses:
            if response['question_id'] == question_to_change:
                target_question = response.copy()
                target_question['correct'] = True  # Flip to correct
                modified_responses.append(target_question)
            else:
                modified_responses.append(response.copy())
        
        if target_question is None:
            return 0  # Question not found
        
        # Calculate new score
        new_total_score, new_module2_difficulties = self.calculate_current_score(modified_responses)
        
        # Calculate impact
        direct_impact = new_total_score - current_total_score
        
        # Check for adaptive penalty changes
        adaptive_penalty_change = 0
        subject = target_question['subject']
        
        if (current_module2_difficulties[subject] != new_module2_difficulties[subject] and 
            target_question['module'] == 1):
            # Module 1 question changed the adaptive path
            adaptive_penalty_change = 50  # Approximate penalty avoidance
        
        return direct_impact + adaptive_penalty_change
    
    def identify_high_impact_questions(self, student_responses: List[Dict], 
                                     top_n: int = 5) -> Dict[str, List[Dict]]:
        """
        Identify the highest impact questions for each subject
        
        Args:
            student_responses: List of student response dictionaries
            top_n: Number of top questions to return per subject
        
        Returns:
            Dictionary with subject as key and list of high-impact questions as value
        """
        # Calculate current score
        current_score, current_module2_difficulties = self.calculate_current_score(student_responses)
        
        # Find all incorrect questions
        incorrect_questions = [r for r in student_responses if not r['correct']]
        
        # Calculate impact for each incorrect question
        question_impacts = []
        
        for question in incorrect_questions:
            impact = self.calculate_impact_score(
                student_responses, 
                question['question_id'], 
                current_score,
                current_module2_difficulties
            )
            
            question_impacts.append({
                'question_id': question['question_id'],
                'subject': question['subject'],
                'module': question['module'],
                'difficulty': question['difficulty'],
                'impact_score': impact,
                'is_module1': question['module'] == 1
            })
        
        # Sort by impact score (descending)
        question_impacts.sort(key=lambda x: x['impact_score'], reverse=True)
        
        # Group by subject and get top N for each
        results = {}
        for subject in self.subjects:
            subject_questions = [q for q in question_impacts if q['subject'] == subject]
            results[subject] = subject_questions[:top_n]
        
        return results
    
    def generate_recommendations(self, student_responses: List[Dict], 
                               top_n: int = 5) -> Dict:
        """
        Generate comprehensive recommendations with explanations
        
        Args:
            student_responses: List of student response dictionaries
            top_n: Number of recommendations per subject
        
        Returns:
            Dictionary containing recommendations and analysis
        """
        current_score, current_module2_difficulties = self.calculate_current_score(student_responses)
        high_impact_questions = self.identify_high_impact_questions(student_responses, top_n)
        
        recommendations = {
            'current_total_score': current_score,
            'current_module2_difficulties': current_module2_difficulties,
            'recommendations': {},
            'summary': {}
        }
        
        for subject in self.subjects:
            subject_questions = high_impact_questions[subject]
            
            # Calculate potential score improvement
            total_potential_gain = sum(q['impact_score'] for q in subject_questions)
            
            # Prioritize Module 1 questions for adaptive benefits
            module1_questions = [q for q in subject_questions if q['module'] == 1]
            
            recommendations['recommendations'][subject] = {
                'high_impact_questions': subject_questions,
                'total_potential_gain': total_potential_gain,
                'module1_priority_count': len(module1_questions)
            }
            
            # Generate summary insights
            if subject_questions:
                avg_impact = total_potential_gain / len(subject_questions)
                recommendations['summary'][subject] = {
                    'average_impact_per_question': avg_impact,
                    'highest_single_impact': subject_questions[0]['impact_score'],
                    'focus_on_module1': len(module1_questions) > len(subject_questions) // 2
                }
        
        return recommendations

# Example usage function
def analyze_student_performance(scoring_data: Dict, student_responses: List[Dict]) -> Dict:
    """
    Main function to analyze student performance and generate recommendations
    
    Args:
        scoring_data: Your JSON scoring data
        student_responses: Student's response data
    
    Returns:
        Comprehensive analysis and recommendations
    """
    analyzer = DSATWhatIfAnalyzer(scoring_data)
    return analyzer.generate_recommendations(student_responses)


# Example of how to structure student response data
def get_student_data(student_id,practicesetId):
    

    client = MongoClient("mongodb://localhost:27017/")

    # 📂 Database & collection
    db = client["test_db"]
    collection = db["student_response"]

    # 📥 Get unique (student_id, practicesetId)
    cursor = collection.find({"student_id": student_id, "practicesetId": practicesetId})
    data = list(cursor)

    transformed_data = []
    subject_counter = {}

    difficulty_map = {
        "easy": "easy",
        "moderate": "medium",
        "hard": "hard"
    }

    for entry in data:
        subject = entry.get("subject", {}).get("name", "Unknown")
        section = entry.get("section", "").lower()
        compleixty = entry.get("compleixty", "moderate").lower()
        topic = entry.get("topic", {}).get("name", "Unknown")

        # Determine module number
        module = 1 if section == "static" else 2

        # Auto-incrementing question ID per (subject, module)
        key = (subject.lower(), module)
        subject_counter[key] = subject_counter.get(key, 0) + 1
        question_id = f"{subject.lower()}_m{module}_q{subject_counter[key]}"

        transformed = {
            "question_id": question_id,
            "subject": subject,
            "module": module,
            "correct": bool(entry.get("correct", 0)),
            "difficulty": difficulty_map.get(compleixty, "medium"),
            "topic": topic
        }

        transformed_data.append(transformed)

    return transformed_data

def get_scoring_map(): 
    client = MongoClient("mongodb://localhost:27017/")
    # 📂 Database & collection
    db = client["test_db"]
    collection = db["scoring_DSAT"]

    # 📥 Get unique (student_id, practicesetId)
    cursor = collection.find()
    data = list(cursor)

    scoring_map = {}
    for val in data:
        scoring_map[val['key']] = {"map":val["map"]}
    return scoring_map
    
print(analyze_student_performance(get_scoring_map(),get_student_data("65aafd6d9acfd21d1abbfaae","66fece275a916f0bb5aea97e")))