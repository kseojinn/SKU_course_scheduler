import streamlit as st
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import List, Dict, Set, Tuple
import random
import copy
import os

# 시간 슬롯 정의 (1교시~15교시)
TIME_SLOTS = {
    1: "09:00~09:50",
    2: "09:55~10:45",
    3: "10:50~11:40",
    4: "11:55~12:45",
    5: "12:50~13:40",
    6: "13:45~14:35",
    7: "14:40~15:30",
    8: "15:35~16:25",
    9: "16:30~17:20",
    10: "17:40~18:30",
    11: "18:30~19:20",
    12: "19:20~20:10",
    13: "20:15~21:05",
    14: "21:05~21:55",
    15: "21:55~22:45"
}

DAYS = ["월", "화", "수", "목", "금", "토", "일"]

@dataclass
class Course:
    """강의 정보 클래스"""
    학년: int
    이수구분: str
    과목코드: str
    분반: str
    교과목명: str
    학점: int
    교수명: str
    수업시간: str
    강의실: str = "-"
    학과: str = ""
    
    def get_display_name(self) -> str:
        """화면 표시용 이름 (교과목명 + 시간 + 교수명)"""
        if self.학과:
            return f"{self.교과목명} [{self.수업시간}] ({self.교수명}) [{self.학과}]"
        return f"{self.교과목명} [{self.수업시간}] ({self.교수명})"
    
    def get_unique_key(self) -> str:
        """고유 식별자 (과목코드 + 분반)"""
        return f"{self.과목코드}-{self.분반}"
    
    def get_time_slots(self) -> List[Tuple[str, int]]:
        """수업시간을 (요일, 교시) 리스트로 변환"""
        if not self.수업시간 or self.수업시간 == "-":
            return []
        
        slots = []
        parts = self.수업시간.split(',')
        
        for part in parts:
            part = part.strip()
            if not part:
                continue
                
            day = part[0]
            if day not in DAYS:
                continue
                
            # 시간 파싱 (예: 월1-3 -> [(월,1), (월,2), (월,3)])
            time_part = part[1:]
            if '-' in time_part:
                start, end = time_part.split('-')
                for i in range(int(start), int(end) + 1):
                    slots.append((day, i))
            else:
                slots.append((day, int(time_part)))
        
        return slots


class TimetableOptimizer:
    """시간표 최적화 클래스"""
    
    def __init__(self, courses: List[Course], optimization_type: str):
        self.all_courses = courses
        self.optimization_type = optimization_type
        self.mandatory_courses = []
        self.excluded_courses = []
        
    def set_mandatory_courses(self, mandatory_course_objects: List[Course]):
        """필수 수강 과목 설정 (Course 객체 리스트)"""
        self.mandatory_courses = mandatory_course_objects
        
    def set_excluded_courses(self, course_keys: List[str]):
        """제외할 과목 설정 (고유 키 기반)"""
        self.excluded_courses = course_keys
        
    def get_available_courses(self) -> List[Course]:
        """수강 가능한 과목 리스트"""
        return [c for c in self.all_courses if c.get_unique_key() not in self.excluded_courses]
    
    def check_conflict(self, timetable: List[Course]) -> bool:
        """시간표 충돌 확인 (시간 충돌 + 과목코드 중복)"""
        # 시간 충돌 확인
        time_map = {}
        for course in timetable:
            slots = course.get_time_slots()
            for slot in slots:
                if slot in time_map:
                    return True  # 시간 충돌 발생
                time_map[slot] = course
        
        # 과목코드 중복 확인
        course_codes = [c.과목코드 for c in timetable]
        if len(course_codes) != len(set(course_codes)):
            return True  # 과목코드 중복 발생
        
        return False
    
    def calculate_fitness(self, timetable: List[Course]) -> float:
        """적합도 계산"""
        if self.check_conflict(timetable):
            return -1000  # 충돌 시 큰 페널티
        
        if self.optimization_type == "오전 수업 회피":
            return self._fitness_avoid_morning(timetable)
        elif self.optimization_type == "점심시간 확보":
            return self._fitness_lunch_time(timetable)
        elif self.optimization_type == "최대 공강 확보":
            return self._fitness_max_free_time(timetable)
        elif self.optimization_type == "요일 분산":
            return self._fitness_distribute_days(timetable)
        
        return 0
    
    def _fitness_avoid_morning(self, timetable: List[Course]) -> float:
        """오전 수업 회피 (1-3교시 최소화)"""
        score = 0
        for course in timetable:
            slots = course.get_time_slots()
            for day, period in slots:
                if period <= 3:
                    score -= 10  # 오전 수업에 페널티
                else:
                    score += 1
        return score
    
    def _fitness_lunch_time(self, timetable: List[Course]) -> float:
        """점심시간 확보 (4-5교시 최소화 - 11:55~13:40)"""
        score = 0
        lunch_slots = set()
        
        for course in timetable:
            slots = course.get_time_slots()
            for day, period in slots:
                if 4 <= period <= 5:  # 4교시(11:55~12:45), 5교시(12:50~13:40)
                    lunch_slots.add(day)
                    score -= 15  # 점심시간 수업에 큰 페널티
                else:
                    score += 1
        
        # 점심시간이 비어있는 요일 보너스
        weekdays = ["월", "화", "수", "목", "금"]
        free_lunch_days = len([d for d in weekdays if d not in lunch_slots])
        score += free_lunch_days * 20
        return score
    
    def _fitness_max_free_time(self, timetable: List[Course]) -> float:
        """최대 공강 확보 (수업 없는 요일 최대화)"""
        days_with_class = set()
        for course in timetable:
            slots = course.get_time_slots()
            for day, _ in slots:
                days_with_class.add(day)
        
        # 평일 기준으로 공강일 계산
        weekdays = ["월", "화", "수", "목", "금"]
        weekday_with_class = len([d for d in days_with_class if d in weekdays])
        free_days = 5 - weekday_with_class
        score = free_days * 100
        
        # 수업이 있는 날은 수업을 몰아서
        for day in days_with_class:
            if day in weekdays:  # 평일만 계산
                day_periods = set()
                for course in timetable:
                    slots = course.get_time_slots()
                    for d, p in slots:
                        if d == day:
                            day_periods.add(p)
                score += len(day_periods) * 5  # 수업을 몰아서 들으면 보너스
        
        return score
    
    def _fitness_distribute_days(self, timetable: List[Course]) -> float:
        """요일 분산 (수업을 여러 요일에 고르게)"""
        day_count = {day: 0 for day in DAYS}
        
        for course in timetable:
            slots = course.get_time_slots()
            counted_days = set()
            for day, _ in slots:
                if day not in counted_days:
                    day_count[day] += 1
                    counted_days.add(day)
        
        # 표준편차가 작을수록 고르게 분산
        counts = list(day_count.values())
        mean = np.mean(counts)
        std = np.std(counts)
        
        score = 100 - std * 20
        
        # 평일에 수업이 고르게 분산되어 있으면 보너스
        weekday_counts = [day_count[d] for d in ["월", "화", "수", "목", "금"]]
        if all(c > 0 for c in weekday_counts):
            score += 50
        
        return score
    
    def hybrid_algorithm(self, target_credits: int, 
                        ga_population_size: int = 50,
                        ga_generations: int = 100,
                        sa_iterations: int = 2000,
                        sa_initial_temp: float = 500) -> List[Course]:
        """
        하이브리드 알고리즘 (유전 알고리즘 + 시뮬레이티드 어닐링)
        
        1단계: 유전 알고리즘으로 좋은 초기 해 집단 생성
        2단계: 각 해에 대해 시뮬레이티드 어닐링으로 지역 최적화
        3단계: 최상의 해 반환
        """
        
        available_courses = self.get_available_courses()
        
        # === 1단계: 유전 알고리즘으로 다양한 좋은 해 생성 ===
        print("1단계: 유전 알고리즘 실행...")
        
        # 초기 인구 생성
        population = []
        for _ in range(ga_population_size):
            individual = list(self.mandatory_courses)
            remaining_credits = target_credits - sum(c.학점 for c in individual)
            
            candidates = [c for c in available_courses if c not in individual]
            random.shuffle(candidates)
            
            for course in candidates:
                if sum(c.학점 for c in individual) + course.학점 <= target_credits:
                    if not self._would_conflict(individual, course):
                        individual.append(course)
                        
            population.append(individual)
        
        # 유전 알고리즘 진화
        for generation in range(ga_generations):
            # 적합도 평가
            fitness_scores = [(ind, self.calculate_fitness(ind)) for ind in population]
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            
            # 선택 (상위 50%)
            population = [ind for ind, _ in fitness_scores[:ga_population_size // 2]]
            
            # 교차 및 돌연변이
            new_population = list(population)
            
            while len(new_population) < ga_population_size:
                parent1, parent2 = random.sample(population, 2)
                child = self._crossover(parent1, parent2, target_credits)
                
                if random.random() < 0.15:
                    child = self._mutate(child, available_courses, target_credits)
                
                new_population.append(child)
            
            population = new_population
        
        # 상위 해들 선택
        fitness_scores = [(ind, self.calculate_fitness(ind)) for ind in population]
        fitness_scores.sort(key=lambda x: x[1], reverse=True)
        top_solutions = [ind for ind, _ in fitness_scores[:5]]  # 상위 5개
        
        # === 2단계: 각 해에 대해 시뮬레이티드 어닐링으로 지역 최적화 ===
        print("2단계: 시뮬레이티드 어닐링으로 지역 최적화...")
        
        best_overall = None
        best_overall_fitness = float('-inf')
        
        for idx, initial_solution in enumerate(top_solutions):
            print(f"  해 {idx+1}/5 최적화 중...")
            
            current = copy.deepcopy(initial_solution)
            current_fitness = self.calculate_fitness(current)
            
            best_local = copy.deepcopy(current)
            best_local_fitness = current_fitness
            
            temp = sa_initial_temp
            cooling_rate = 0.995
            
            for iteration in range(sa_iterations):
                # 이웃 해 생성
                neighbor = self._get_neighbor(current, available_courses, target_credits)
                neighbor_fitness = self.calculate_fitness(neighbor)
                
                # 수락 여부 결정
                delta = neighbor_fitness - current_fitness
                
                if delta > 0 or random.random() < np.exp(delta / max(temp, 0.01)):
                    current = neighbor
                    current_fitness = neighbor_fitness
                    
                    if current_fitness > best_local_fitness:
                        best_local = copy.deepcopy(current)
                        best_local_fitness = current_fitness
                
                temp *= cooling_rate
            
            # 전체 최고 해 업데이트
            if best_local_fitness > best_overall_fitness:
                best_overall = best_local
                best_overall_fitness = best_local_fitness
        
        print(f"최종 적합도: {best_overall_fitness:.2f}")
        return best_overall
    
    def genetic_algorithm(self, target_credits: int, population_size: int = 100, 
                         generations: int = 200, mutation_rate: float = 0.1) -> List[Course]:
        """유전 알고리즘으로 최적 시간표 생성"""
        
        available_courses = self.get_available_courses()
        
        # 초기 인구 생성
        population = []
        for _ in range(population_size):
            individual = list(self.mandatory_courses)
            remaining_credits = target_credits - sum(c.학점 for c in individual)
            
            candidates = [c for c in available_courses if c not in individual]
            random.shuffle(candidates)
            
            for course in candidates:
                if sum(c.학점 for c in individual) + course.학점 <= target_credits:
                    if not self._would_conflict(individual, course):
                        individual.append(course)
                        
            population.append(individual)
        
        # 진화 과정
        best_individual = None
        best_fitness = float('-inf')
        
        for generation in range(generations):
            # 적합도 평가
            fitness_scores = [(ind, self.calculate_fitness(ind)) for ind in population]
            fitness_scores.sort(key=lambda x: x[1], reverse=True)
            
            if fitness_scores[0][1] > best_fitness:
                best_fitness = fitness_scores[0][1]
                best_individual = copy.deepcopy(fitness_scores[0][0])
            
            # 선택 (상위 50%)
            population = [ind for ind, _ in fitness_scores[:population_size // 2]]
            
            # 교차 및 돌연변이
            new_population = list(population)
            
            while len(new_population) < population_size:
                parent1, parent2 = random.sample(population, 2)
                child = self._crossover(parent1, parent2, target_credits)
                
                if random.random() < mutation_rate:
                    child = self._mutate(child, available_courses, target_credits)
                
                new_population.append(child)
            
            population = new_population
        
        return best_individual
    
    def simulated_annealing(self, target_credits: int, initial_temp: float = 1000, 
                           cooling_rate: float = 0.995, iterations: int = 5000) -> List[Course]:
        """시뮬레이티드 어닐링으로 최적 시간표 생성"""
        
        available_courses = self.get_available_courses()
        
        # 초기 해 생성
        current = list(self.mandatory_courses)
        candidates = [c for c in available_courses if c not in current]
        random.shuffle(candidates)
        
        for course in candidates:
            if sum(c.학점 for c in current) + course.학점 <= target_credits:
                if not self._would_conflict(current, course):
                    current.append(course)
        
        current_fitness = self.calculate_fitness(current)
        best = copy.deepcopy(current)
        best_fitness = current_fitness
        
        temp = initial_temp
        
        for iteration in range(iterations):
            # 이웃 해 생성
            neighbor = self._get_neighbor(current, available_courses, target_credits)
            neighbor_fitness = self.calculate_fitness(neighbor)
            
            # 수락 여부 결정
            delta = neighbor_fitness - current_fitness
            
            if delta > 0 or random.random() < np.exp(delta / temp):
                current = neighbor
                current_fitness = neighbor_fitness
                
                if current_fitness > best_fitness:
                    best = copy.deepcopy(current)
                    best_fitness = current_fitness
            
            temp *= cooling_rate
        
        return best
    
    def _would_conflict(self, timetable: List[Course], new_course: Course) -> bool:
        """새 강의 추가 시 충돌 여부 확인 (시간 충돌 + 과목코드 중복)"""
        # 시간 충돌 확인
        existing_slots = set()
        for course in timetable:
            existing_slots.update(course.get_time_slots())
        
        new_slots = new_course.get_time_slots()
        if existing_slots & set(new_slots):
            return True  # 시간 충돌
        
        # 과목코드 중복 확인
        existing_course_codes = set(c.과목코드 for c in timetable)
        if new_course.과목코드 in existing_course_codes:
            return True  # 과목코드 중복
        
        return False
    
    def _crossover(self, parent1: List[Course], parent2: List[Course], target_credits: int) -> List[Course]:
        """교차 연산"""
        child = list(self.mandatory_courses)
        
        # 부모에서 유전자 선택 (set 대신 고유 키로 중복 제거)
        all_genes_dict = {}
        for course in parent1 + parent2:
            all_genes_dict[course.get_unique_key()] = course
        
        all_genes = list(all_genes_dict.values())
        random.shuffle(all_genes)
        
        for course in all_genes:
            if course in self.mandatory_courses:
                continue
            if sum(c.학점 for c in child) + course.학점 <= target_credits:
                if not self._would_conflict(child, course):
                    child.append(course)
        
        return child
    
    def _mutate(self, individual: List[Course], available_courses: List[Course], 
                target_credits: int) -> List[Course]:
        """돌연변이 연산"""
        mutated = copy.deepcopy(individual)
        
        # 필수 과목이 아닌 과목 중 하나를 제거
        non_mandatory = [c for c in mutated if c not in self.mandatory_courses]
        if non_mandatory:
            mutated.remove(random.choice(non_mandatory))
        
        # 새로운 과목 추가 시도
        candidates = [c for c in available_courses if c not in mutated]
        random.shuffle(candidates)
        
        for course in candidates:
            if sum(c.학점 for c in mutated) + course.학점 <= target_credits:
                if not self._would_conflict(mutated, course):
                    mutated.append(course)
                    break
        
        return mutated
    
    def _get_neighbor(self, current: List[Course], available_courses: List[Course], 
                     target_credits: int) -> List[Course]:
        """이웃 해 생성"""
        neighbor = copy.deepcopy(current)
        
        # 필수 과목이 아닌 과목 중 하나를 교체
        non_mandatory = [c for c in neighbor if c not in self.mandatory_courses]
        
        if non_mandatory and random.random() < 0.7:
            # 과목 교체
            neighbor.remove(random.choice(non_mandatory))
            
            candidates = [c for c in available_courses if c not in neighbor]
            random.shuffle(candidates)
            
            for course in candidates:
                if sum(c.학점 for c in neighbor) + course.학점 <= target_credits:
                    if not self._would_conflict(neighbor, course):
                        neighbor.append(course)
                        break
        else:
            # 과목 추가 또는 제거
            if random.random() < 0.5 and non_mandatory:
                neighbor.remove(random.choice(non_mandatory))
            else:
                candidates = [c for c in available_courses if c not in neighbor]
                random.shuffle(candidates)
                
                for course in candidates:
                    if sum(c.학점 for c in neighbor) + course.학점 <= target_credits:
                        if not self._would_conflict(neighbor, course):
                            neighbor.append(course)
                            break
        
        return neighbor


def load_course_data(department_file: str, department_name: str = "") -> List[Course]:
    """JSON 파일에서 강의 데이터 로드"""
    with open(department_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    courses = []
    course_list = data.get('course', [])
    
    for item in course_list:
        try:
            course = Course(
                학년=int(item.get('grade', 0)),
                이수구분=item.get('type', ''),
                과목코드=item.get('id', ''),
                분반=item.get('section', ''),
                교과목명=item.get('name', ''),
                학점=int(item.get('credits', 0)),
                교수명=item.get('professor', ''),
                수업시간=item.get('time', ''),
                강의실='-',
                학과=department_name
            )
            courses.append(course)
        except Exception as e:
            continue  # 잘못된 데이터는 스킵
    
    return courses


def create_course_selector(courses: List[Course], label: str, key: str, 
                           excluded_keys: List[str] = None) -> List[str]:
    """
    검색 가능한 과목 선택 위젯 생성
    
    Args:
        courses: 전체 과목 리스트
        label: 위젯 라벨
        key: 위젯 고유 키
        excluded_keys: 제외할 과목의 고유 키 리스트
    
    Returns:
        선택된 과목의 고유 키 리스트
    """
    if excluded_keys is None:
        excluded_keys = []
    
    # 제외된 과목을 필터링
    available_courses = [c for c in courses if c.get_unique_key() not in excluded_keys]
    
    # 과목 표시명과 고유 키 매핑
    course_options = {c.get_display_name(): c.get_unique_key() for c in available_courses}
    
    # 검색 가능한 multiselect (모든 과목 표시)
    selected_display_names = st.multiselect(
        label,
        options=list(course_options.keys()),
        key=key,
        help="타이핑으로 과목을 검색할 수 있습니다"
    )
    
    # 선택된 과목의 고유 키 반환
    selected_keys = [course_options[name] for name in selected_display_names]
    
    return selected_keys


def display_timetable(timetable: List[Course]):
    """시간표를 표 형식으로 출력"""
    # 시간표 그리드 생성
    grid = {day: {period: [] for period in range(1, 16)} for day in DAYS}
    
    for course in timetable:
        slots = course.get_time_slots()
        for day, period in slots:
            if period <= 15:  # 15교시까지만 표시
                grid[day][period].append(f"{course.교과목명}\n({course.교수명})")
    
    # DataFrame 생성
    df_data = []
    for period in range(1, 16):
        row = [TIME_SLOTS[period]]
        for day in DAYS:
            cell_content = "\n\n".join(grid[day][period]) if grid[day][period] else ""
            row.append(cell_content)
        df_data.append(row)
    
    df = pd.DataFrame(df_data, columns=["시간"] + DAYS)
    
    st.dataframe(df, use_container_width=True, height=800)


def main():
    st.set_page_config(page_title="성결대 시간표 최적화", layout="wide")
    
    st.title("🎓 성결대학교 시간표 최적화 시스템")
    st.markdown("---")
    
    # 사이드바 설정
    with st.sidebar:
        st.header("⚙️ 설정")
        
        # 학과 선택 (파이데이아 제외)
        department_files = {
            "신학과": "theology.json",
            "기독교교육상담학과": "christian_education_and_counseling.json",
            "문화선교학과": "cultural_mission.json",
            "국어국문학과": "korean_language_and_literature.json",
            "영어영문학과": "english_language_and_literature.json",
            "중어중문학과": "chinese_language_and_literature.json",
            "관광학과": "tourism.json",
            "국제개발협력학과": "international_development_and_cooperation.json",
            "사회복지학과": "social_welfare.json",
            "행정학부": "public_administration.json",
            "경영학과": "business_administration.json",
            "글로벌물류학과": "global_logistics.json",
            "산업경영공학과": "industrial_engineering.json",
            "유아교육과": "early_childhood_education.json",
            "체육교육과": "physical_education.json",
            "컴퓨터공학과": "computer_engineering.json",
            "정보통신공학과": "information_and_communication_engineering.json",
            "미디어소프트웨어학과": "media_software.json",
            "도시디자인정보공학과": "urban_design_and_information_engineering.json",
            "음악학부": "music.json",
            "연극영화학부": "theater_and_film.json",
            "뷰티디자인학과": "beauty_design.json",
            "실용음악과": "practical_music.json"
        }
        
        st.subheader("📂 데이터 경로 설정")
        data_path = st.text_input(
            "강의 데이터 폴더 경로",
            value=r"C:\"
        )
        
        selected_department = st.selectbox(
            "학과 선택",
            options=list(department_files.keys())
        )
        
        st.subheader("🎯 최적화 유형")
        optimization_type = st.selectbox(
            "시간표 유형",
            options=["오전 수업 회피", "점심시간 확보", "최대 공강 확보", "요일 분산"]
        )
        
        st.subheader("🧬 알고리즘 선택")
        algorithm = st.selectbox(
            "최적화 알고리즘",
            options=["하이브리드", "유전 알고리즘", "시뮬레이티드 어닐링"]
        )
        
        st.subheader("📊 목표 학점")
        target_credits = st.slider("목표 학점", min_value=12, max_value=24, value=18, step=1)
    
    # 메인 영역
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("📝 수강 설정")
        
        # 선택한 학과 데이터 로드
        major_file_path = os.path.join(data_path, department_files[selected_department])
        paideia_file_path = os.path.join(data_path, "paideia.json")
        
        if os.path.exists(major_file_path):
            # 선택한 학과 강의 로드
            major_courses = load_course_data(major_file_path, selected_department)
            st.success(f"✅ {selected_department}: {len(major_courses)}개 강의")
            
            # 파이데이아 강의 로드
            paideia_courses = []
            if os.path.exists(paideia_file_path):
                paideia_courses = load_course_data(paideia_file_path, "파이데이아학부")
                st.success(f"✅ 파이데이아학부(교양): {len(paideia_courses)}개 강의")
            
            # 모든 학과 강의 로드 (필수 수강 과목 선택용)
            st.info("전체 학과 강의 로딩 중...")
            all_department_courses = []
            for dept_name, dept_file in department_files.items():
                dept_path = os.path.join(data_path, dept_file)
                if os.path.exists(dept_path):
                    dept_courses = load_course_data(dept_path, dept_name)
                    all_department_courses.extend(dept_courses)
            
            st.success(f"✅ 전체 {len(all_department_courses)}개 전공 과목 로드 완료")
            
            st.markdown("---")
            
            # === 이미 수강한 과목 선택 ===
            st.markdown("### 🚫 이미 수강한 과목")
            st.caption("💡 선택한 학과 + 교양 과목에서 선택")
            st.caption("⚠️ 선택한 과목과 같은 과목코드를 가진 모든 분반이 자동으로 제외됩니다")
            
            # 선택한 학과 + 파이데이아 과목
            courses_for_exclusion = major_courses + paideia_courses
            
            excluded_keys = create_course_selector(
                courses=courses_for_exclusion,
                label="제외할 과목 선택 (타이핑으로 검색 가능)",
                key="excluded"
            )
            
            # 선택한 과목들의 과목코드 수집 및 같은 코드 모두 제외
            excluded_course_codes = set()
            for course in courses_for_exclusion:
                if course.get_unique_key() in excluded_keys:
                    excluded_course_codes.add(course.과목코드)
            
            # 같은 과목코드를 가진 모든 과목을 제외 리스트에 추가
            all_excluded_keys = list(excluded_keys)
            for course in courses_for_exclusion:
                if course.과목코드 in excluded_course_codes:
                    if course.get_unique_key() not in all_excluded_keys:
                        all_excluded_keys.append(course.get_unique_key())
            
            if excluded_course_codes:
                st.info(f"🔒 제외된 과목코드: {', '.join(sorted(excluded_course_codes))}")
                st.info(f"📊 총 {len(all_excluded_keys)}개 분반 제외됨")
            
            st.markdown("---")
            
            # === 필수 수강 과목 선택 ===
            st.markdown("### ✅ 필수 수강 과목")
            st.caption("💡 모든 학과의 전공 과목 + 교양 과목에서 선택 가능")
            st.caption("🎯 선택한 필수 과목 + 나머지는 교양 과목으로 채워집니다")
            
            # 전공 + 교양 모두 합치기
            all_courses_with_paideia = all_department_courses + paideia_courses
            
            mandatory_keys = create_course_selector(
                courses=all_courses_with_paideia,
                label="필수 수강 과목 선택 (타이핑으로 검색 가능)",
                key="mandatory",
                excluded_keys=[]  # 필수 과목은 제외 과목과 무관하게 선택 가능
            )
            
            # 필수 과목 객체 찾기
            mandatory_course_objects = [c for c in all_courses_with_paideia if c.get_unique_key() in mandatory_keys]
            
            if mandatory_course_objects:
                mandatory_credits = sum(c.학점 for c in mandatory_course_objects)
                mandatory_major_count = len([c for c in mandatory_course_objects if c.학과 != "파이데이아학부"])
                mandatory_paideia_count = len([c for c in mandatory_course_objects if c.학과 == "파이데이아학부"])
                
                # 과목코드 중복 확인
                mandatory_course_codes = [c.과목코드 for c in mandatory_course_objects]
                if len(mandatory_course_codes) != len(set(mandatory_course_codes)):
                    st.warning("⚠️ 필수 과목에 같은 과목코드가 중복되었습니다! 시간표 생성이 실패할 수 있습니다.")
                    # 중복된 과목코드 표시
                    duplicates = [code for code in set(mandatory_course_codes) if mandatory_course_codes.count(code) > 1]
                    st.error(f"🔴 중복된 과목코드: {', '.join(duplicates)}")
                else:
                    st.info(f"📚 필수 과목: 전공 {mandatory_major_count}개 + 교양 {mandatory_paideia_count}개 = 총 {mandatory_credits}학점")
            
            st.markdown("---")
            
            # 최적화 실행 버튼
            if st.button("🚀 시간표 생성", type="primary", use_container_width=True):
                if not paideia_courses:
                    st.error("❌ 파이데이아학부(교양) 데이터를 찾을 수 없습니다!")
                else:
                    with st.spinner("최적 시간표를 생성하는 중..."):
                        # 최적화에 사용할 과목: 파이데이아(교양) 과목만 (제외 과목 제거)
                        available_paideia = [c for c in paideia_courses if c.get_unique_key() not in all_excluded_keys]
                        
                        # 최적화 실행
                        optimizer = TimetableOptimizer(available_paideia, optimization_type)
                        optimizer.set_excluded_courses(all_excluded_keys)
                        optimizer.set_mandatory_courses(mandatory_course_objects)
                        
                        if algorithm == "하이브리드":
                            best_timetable = optimizer.hybrid_algorithm(target_credits)
                        elif algorithm == "유전 알고리즘":
                            best_timetable = optimizer.genetic_algorithm(target_credits)
                        else:  # 시뮬레이티드 어닐링
                            best_timetable = optimizer.simulated_annealing(target_credits)
                        
                        # 결과 저장
                        st.session_state['best_timetable'] = best_timetable
                        st.session_state['fitness'] = optimizer.calculate_fitness(best_timetable)
                        st.session_state['mandatory_course_objects'] = mandatory_course_objects  # 필수 과목 저장
                        
                        st.success("✅ 시간표 생성 완료!")
            
        else:
            st.error(f"❌ 파일을 찾을 수 없습니다: {major_file_path}")
    
    with col2:
        st.subheader("📅 최적화된 시간표")
        
        if 'best_timetable' in st.session_state:
            timetable = st.session_state['best_timetable']
            fitness = st.session_state['fitness']
            
            # 요약 정보
            total_credits = sum(c.학점 for c in timetable)
            
            col_a, col_b = st.columns(2)
            with col_a:
                st.metric("총 학점", f"{total_credits}학점")
            with col_b:
                st.metric("적합도 점수", f"{fitness:.2f}")
            
            # 과목 분류 - mandatory_course_objects를 session_state에서 가져오기
            if 'mandatory_course_objects' in st.session_state:
                mandatory_objs = st.session_state['mandatory_course_objects']
                mandatory_keys_in_timetable = [c.get_unique_key() for c in mandatory_objs]
                
                mandatory_major_count = len([c for c in timetable 
                                            if c.get_unique_key() in mandatory_keys_in_timetable 
                                            and c.학과 != "파이데이아학부"])
                mandatory_paideia_count = len([c for c in timetable 
                                              if c.get_unique_key() in mandatory_keys_in_timetable 
                                              and c.학과 == "파이데이아학부"])
                auto_paideia_count = len([c for c in timetable 
                                         if c.get_unique_key() not in mandatory_keys_in_timetable 
                                         and c.학과 == "파이데이아학부"])
                
                st.info(f"📊 필수 전공: {mandatory_major_count}개 | 필수 교양: {mandatory_paideia_count}개 | 자동 교양: {auto_paideia_count}개")
            else:
                paideia_count = len([c for c in timetable if c.학과 == "파이데이아학부"])
                st.info(f"📊 교양: {paideia_count}개")
            
            # 시간표 표시
            display_timetable(timetable)
            
            st.markdown("---")
            
            # 수강 과목 목록
            st.subheader("📚 수강 과목 목록")
            
            course_df = pd.DataFrame([
                {
                    "교과목명": c.교과목명,
                    "학점": c.학점,
                    "교수명": c.교수명,
                    "수업시간": c.수업시간,
                    "이수구분": c.이수구분,
                    "분반": c.분반,
                    "학과": c.학과
                }
                for c in timetable
            ])
            
            st.dataframe(course_df, use_container_width=True)
            
        else:
            st.info("👆 왼쪽에서 설정을 완료하고 '시간표 생성' 버튼을 클릭하세요.")
            
            st.markdown("""
            ### 📖 사용 방법
            
            1. **학과 선택**: 자신의 전공 학과를 선택하세요
            2. **제외할 과목**: 이미 수강한 과목을 선택하세요 (같은 과목코드의 모든 분반이 자동 제외됩니다)
            3. **필수 수강 과목**: 꼭 들어야 하는 전공 과목 및 교양 과목을 선택하세요
            4. **최적화 유형**: 원하는 시간표 스타일을 선택하세요
            5. **시간표 생성**: 버튼을 클릭하면 자동으로 교양 과목이 채워진 시간표가 생성됩니다!
            
            ✨ **시간표 구성**: 필수 과목 (전공 + 교양) + 나머지는 교양 과목으로 자동 채움
            """)



if __name__ == "__main__":
    main()
