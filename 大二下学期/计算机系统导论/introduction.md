# 计算机系统导论 (CSAPP 15213) 课程信息与学习建议

> **课程名称**：计算机系统导论  
> **英文名称**：Introduction to Computer Systems  
> **课程代码**：SAI12003  
> **课程性质**：专业必修课（专业基础课）  
> **学分数**：4 学分  
> **适用专业**：计算机科学与技术、人工智能、数据科学与大数据技术等  
> **授课语言**：中文授课  
> **学时数**：64  
> **开设学期**：春季  
> **建议最大选课人数**：200  
> **授课教师**：计卫星 教授  
> **联系方式**：Email: jwx@bit.edu.cn  
> **先修课程**：计算机科学导论、程序设计基础、数字逻辑

---

## 一、课程简介

本课程将计算机系统中高级语言程序与数字逻辑电路之间的相关知识点有机关联，构建出完整的计算机系统基本框架，帮助大家建立整机概念，强化“系统思维”。  
同时，课程还会介绍国产自主可控计算机系统建设情况，鼓励大家投身国产软件系统研发，培养担当精神和责任意识，为国家科技战略贡献力量。

---

## 二、课程目标

- **教学目标**：  
  使学生能够从程序员角度认识计算机系统，理解高级语言程序、操作系统、编译器、链接器、指令集体系结构及微架构等核心内容之间的关联；提升程序调试、性能优化、移植与健壮性设计能力，为后续《计算机组成与系统结构》、《操作系统》和《编译技术》等课程打下坚实基础。

- **育人目标**：  
  通过介绍国产自主可控系统建设，激发大家的责任感与使命感，将个人发展与国家科技战略结合，培养创新精神和团队协作能力。

---

## 三、教学内容和学时分配

以下内容结合了课程大纲与授课目录，具体包括：

1. **计算机系统概论** (2 学时)  
   - **主要内容**：编译系统的地位与作用、处理器及存储系统层次结构、操作系统与设备管理、网络通信。  
   - **教学要求**：了解计算机系统的基本构成、工作流程及各子系统之间的接口关系。  
   - **重点难点**：各抽象层次之间的接口与交互。

2. **信息的表示和处理** (6 学时)  
   - **主要内容**：信息的位级表示与存储、位级运算、整数编码与运算、浮点数表示与运算。  
   - **教学要求**：掌握主要数据类型的表示方法，理解数据编码格式对计算方式和精度的影响。  
   - **实践环节**：Lab 1 (data)。

3. **程序的机器级表示** (10 学时)  
   - **主要内容**：基于 Intel 及国产处理器的程序机器级表示，涉及内存越界与溢出问题。  
   - **教学要求**：掌握汇编语言程序的整体结构、数据表示与访问、控制流、过程调用、数组及复合数据结构表示。  
   - **重点难点**：精简指令集与复杂指令集之间的表示差异。  
   - **实践环节**：Lab 2 (Bomb)、Lab 3 (Attack)。

4. **处理器体系结构与存储系统** (12 学时)  
   - **主要内容**：指令流水线、超标量与乱序执行、分支预测、投机执行、Cache 系统与虚拟存储系统。  
   - **教学要求**：了解当代处理器微架构设计及优化技术，理解机器代码在处理器中的执行过程。  
   - **重点难点**：处理器微体系结构对程序运行效率的影响。

5. **程序性能优化** (10 学时)  
   - **主要内容**：编译器优化及其局限、循环展开、指令级并行优化及其他与处理器微架构相关的优化。  
   - **教学要求**：理解编译优化原理及其局限性，能运用所学知识编写高效程序。  
   - **重点难点**：面向现代高性能处理器的性能优化策略。

6. **程序链接** (8 学时)  
   - **主要内容**：可执行文件格式、静态链接、动态链接、可执行文件加载过程。  
   - **教学要求**：掌握可执行文件生成、静态与动态链接及加载过程，重点关注位置无关代码的链接。  
   - **实践环节**：Lab 4 (Cache)、Lab 5 (Shell)。

7. **虚拟内存** (6 学时)  
   - **主要内容**：虚拟内存的地位、作用、与缓存的关系、地址翻译、动态内存分配与管理。  
   - **教学要求**：理解虚拟内存的工作原理及地址翻译过程，掌握动态内存管理机制。  
   - **重点难点**：带 Cache 的虚拟内存地址翻译及数据访问过程。  
   - **实践环节**：Lab 6 (Malloc)。

8. **异常处理** (6 学时)  
   - **主要内容**：异常及其分类、进程控制、信号、非本地跳转。  
   - **教学要求**：理解进程运行动态、常见异常及其处理、信号机制。  
   - **重点难点**：异常响应与处理流程。

9. **并发编程** (4 学时)  
   - **主要内容**：进程级与线程级并发、并发同步及相关问题。  
   - **教学要求**：掌握并发编程基本方法及同步技术，理解并发问题的成因及预防措施。  
   - **重点难点**：并发问题的产生及解决方法。

---

## 四、课程评分及考核方式

- **总分构成**：平时与期末 4:6  
  - **平时成绩**：  
    - 主要依据实验与签到。  
    - 实验要求严格，完成所有实验（包括上机报告详细截图、心得说明、SQL 代码需以代码块形式呈现）即能获得满分。  
  - **期末考试**：  
    - 形式：选择题（单选题40道，共40分；多选题20道，共40分）+ 综合题（2道，共20分）。  
    - 考试题目难度较低，但覆盖面广，需全面复习。  
    - **注意**：2024 春季期末综合题中，第1题为典型虚拟内存及地址翻译题（与 slides 例题几乎一致），第2题考查性能优化问题。

- **录入成绩**：较慢

---

## 五、课程学习建议

- **平时听课**：本课程老师讲解非常清晰，只要平时听懂基本内容即可，但不要忽视课堂作业与实验要求。
- **课前预习与课后复习**：  
  - 强烈建议大家在初期认真阅读教材《深入理解计算机系统》（中文版或英文版均可），并结合 CSAPP 网课辅助理解（网课仅为辅助，不是重点）。
  - 期末复习时，重点关注 lecture slides，特别是虚拟内存、程序性能优化、关系型数据表示及异常处理等部分。
- **实践为主**：实验环节内容丰富，切勿马虎，每次上机实验报告都要求详尽记录、截图及心得体会。

- **相关材料**：  
  - 附带一套北理工往年由计卫星教授出的题目，供大家参考练习。

---

# Introduction to Computer Systems (CSAPP 15213) – Course Overview & Study Tips

> **Course Name**: Introduction to Computer Systems  
> **Course Code**: SAI12003  
> **Course Type**: Required Core Course  
> **Credits**: 4  
> **Applicable Majors**: Computer Science, Artificial Intelligence, Data Science & Big Data Technology, etc.  
> **Instruction Language**: Chinese  
> **Total Hours**: 64  
> **Semester Offered**: Spring  
> **Instructor**: Prof. Ji Weixing (Email: jwx@bit.edu.cn)  
> **Prerequisites**: Introduction to Computer Science, Programming Fundamentals, Digital Logic

---

## 1. Course Introduction

This course bridges the gap between high-level programming and digital logic circuits, building a complete framework for understanding computer systems. Students will develop a comprehensive concept of how computer systems work and enhance their "systems thinking."  
In addition, the course introduces the development of domestically controllable computer systems, encouraging students to engage in research and development of indigenous software systems and cultivate a sense of responsibility and mission.

---

## 2. Course Objectives

- **Educational Objectives**:  
  Enable students to view computer systems from a programmer’s perspective, understanding the interconnections among high-level language programs, operating systems, compilers, linkers, instruction set architectures, and microarchitectures. This will enhance skills in debugging, performance optimization, porting, and robust programming, laying a solid foundation for courses in Computer Organization, Operating Systems, and Compiler Design.

- **Holistic Goals**:  
  Through discussions on domestically controllable systems, students are encouraged to take responsibility and develop a strong sense of mission, integrating personal skills with national technological strategies.

---

## 3. Syllabus Overview & Contact Hours

1. **Introduction to Computer Systems** (2 hours)  
   - Topics: Role of compilers, processor and memory hierarchy, OS and device management, network communication.  
   - Aim: Understand system components and their interactions.  
   - Focus: Interfaces among different abstraction layers.

2. **Representation and Processing of Information** (6 hours)  
   - Topics: Bit-level representation, bitwise operations, integer and floating-point encoding and arithmetic.  
   - Aim: Master data representation methods and understand the impact on range, computation, and precision.  
   - Lab: Data representation exercise.

3. **Machine-Level Representation of Programs** (10 hours)  
   - Topics: Machine-level representation on Intel and domestic processors; issues of memory overflow and boundary errors.  
   - Aim: Grasp assembly language structure, data representation, data access, control flow, procedure calls, and composite data structures.  
   - Focus: Differences between RISC and CISC assembly representations.  
   - Labs: Bomb exercise, Attack exercise.

4. **Processor Architecture and Memory Systems** (12 hours)  
   - Topics: Instruction pipelining, superscalar, out-of-order execution, branch prediction, speculative execution, Cache, and virtual memory systems.  
   - Aim: Understand modern processor microarchitecture design and how machine code is executed.  
   - Focus: Impact of microarchitecture on program performance.

5. **Program Performance Optimization** (10 hours)  
   - Topics: Compiler optimizations and limitations, loop unrolling, instruction-level parallelism, and other microarchitecture-related optimizations.  
   - Aim: Understand compiler optimization principles and apply them to write efficient programs.  
   - Focus: Optimization techniques for high-performance processors.

6. **Program Linking** (8 hours)  
   - Topics: Executable file formats, static and dynamic linking, loading processes.  
   - Aim: Understand how executables are created and loaded, with an emphasis on position-independent code.  
   - Labs: Cache exercise, Shell exercise.

7. **Virtual Memory** (6 hours)  
   - Topics: Role and function of virtual memory, its relationship with cache, address translation, dynamic memory allocation, and management.  
   - Aim: Master the principles of virtual memory systems and the translation from virtual to physical addresses.  
   - Focus: Virtual memory with cache and data access.  
   - Lab: Malloc exercise.

8. **Exception Handling** (6 hours)  
   - Topics: Classification of exceptions, process control, signals, non-local jumps.  
   - Aim: Understand process dynamics, common exceptions and their handling, and signal mechanisms.  
   - Focus: Response and handling procedures.

9. **Concurrent Programming** (4 hours)  
   - Topics: Process-level and thread-level concurrency, synchronization, and related issues.  
   - Aim: Learn fundamental methods of concurrent programming and synchronization techniques, and understand how to prevent concurrency issues.
 
---

## 4. Evaluation & Grading

- **Grading Ratio**: Coursework : Final Exam = 4 : 6  
  - **Coursework**:  
    - Based on lab performance and attendance.  
    - Complete all labs and activities (including detailed lab reports with screenshots, personal reflections, and SQL code presented as code blocks) to secure full marks.
  - **Final Exam**:  
    - Composed of multiple-choice questions (40 single-choice questions for 40 points, 20 multiple-choice questions for 40 points) and 2 comprehensive questions (20 points total).  
    - Exam questions are of low difficulty but cover a broad range of topics.  
    - **Note**: In Spring 2024, the first comprehensive question was a typical virtual memory and address translation problem (almost identical to the slide example), and the second focused on performance optimization.

- **Grade Posting**: Grades are entered slowly.

---

## 5. Study Tips

- **Class Participation**:  
  Although the lectures are generally clear, do not miss classes since assignments and labs are crucial to your coursework score.

- **Pre-study & Review**:  
  - It is strongly recommended to read the textbook *深入理解计算机系统* (either Chinese or English) early on, and optionally supplement with CSAPP online courses (as an aid, not the primary resource).  
  - Focus your review on the lecture slides, especially topics such as virtual memory, performance optimization, machine-level program representation, and exception handling.

- **Practical Emphasis**:  
  The course includes numerous labs and practice sessions. Ensure that you complete all lab reports with detailed screenshots, personal insights, and proper formatting for code segments.

- **Additional Resources**:  
  A set of past exam questions compiled by Prof. Ji Weixing (from BIT) is provided for further practice.

---

