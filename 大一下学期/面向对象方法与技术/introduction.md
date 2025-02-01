# 面向对象方法与技术 课程信息与学习建议
**英文名称**: Object-oriented Method and Technology

---

## 一、课程基本信息

- **课程代码**: SAI11001 / BQ10214401 (可选)
- **课程性质**: 专业必修课（专业基础课、专业核心课）；学科基础课程
- **学分数**: 2
- **适用专业**: 计算机科学与技术、人工智能
- **授课语言**: 中文授课
- **学时数**: 64
- **开设学期**: 春季

- **授课教师**: 申佳丽  
- **教师职称**: 讲师  
- **联系方式**:  
  - Email: sjl@bnu.edu.cn  

- **先修课程要求**: 程序设计基础

---

## 二、课程简介

《面向对象方法与技术》以 C++ 为主要编程语言。  
C++ 由 Bjarne Stroustrup 于 1983 年开发，是 C 语言的扩展，它不仅引入了面向对象的编程方法，还通过模板技术支持泛型编程，能够实现高度可重用的代码和算法。  
课程内容涵盖面向对象程序设计的基本概念、类和对象、数据共享与保护、数组、指针与字符串、类的继承、多态性、模板及 STL、流类库、异常处理以及 Qt 程序设计基础。  
总体来说，C++ 语言结合了面向过程和面向对象的思想，具有高效、灵活、可重用等优势，本课程旨在帮助同学们掌握 C++ 基础知识和应用能力，并为后续课程（如数据结构、算法分析、计算机视觉、图形学等）打下坚实基础。

---

## 三、课程目标

### 教学目标
- **理论与实践结合**：  
  通过理论学习和编程实验，使学生掌握 C++ 编程语言的基本语法、面向对象的设计思想以及泛型编程方法。
  
- **知识应用**：  
  能够将所学知识应用于计算机程序设计、数据结构与算法分析、计算机视觉、图形学等领域，培养程序设计与系统开发能力。

### 育人目标
- 培养学生勤奋踏实的学习态度和不畏困难的精神；  
- 激发探索新知识、新技术的积极性；  
- 提高团队协作意识，增强综合解决问题的能力。

---

## 四、教学内容与学时分配

### （一）绪论 – 4 学时  
*（2 课堂讲授学时 + 2 实践学时）*  
**主要内容**:  
- 面向对象程序设计（OOP）的基本概念  
- 面向过程与面向对象的区别  
- OOP 发展史及其特征、优点  
- C++ 对 C 的扩充与泛型程序设计  
- C++ 程序简介、数据输入输出及开发工具  
**教学要求**:  
- 掌握面向对象程序设计的特征和优点，了解 C++ 程序开发流程。  
**实践活动**:  
- 安装 C++ 开发工具；编写 “Hello World！” 程序；实现简单的学生信息输入与输出。

---

### （二）第一章：类和对象 – 8 学时  
*（4 课堂讲授学时 + 4 实践学时）*  
**主要内容**:  
- 类的概念与设计  
- 成员的访问修饰符：private/protected/public  
- 数据成员与成员函数，封装性  
- 对象的构造与析构，this 指针  
- 类的组合与前向引用声明  
**教学要求**:  
- 掌握类的设计、构造函数和析构函数的使用。  
**实践活动**:  
- 编写矩形类、学生类、包含生日信息的学生类等。

---

### （三）第二章：数据的共享与保护 – 类的静态成员、友元与常成员 – 8 学时  
*（4 课堂讲授学时 + 4 实践学时）*  
**主要内容**:  
- 作用域与生命周期  
- 静态成员（static）的定义与使用  
- 友元（friend）的概念  
- 常成员与常对象（const）的使用  
**教学要求**:  
- 掌握静态成员、友元和常成员的设计和使用，了解数据共享与保护以及多文件程序组织。  
**实践活动**:  
- 编写鱼类、学生自动编号、储蓄存款类、家庭电量统计、const 函数参数示例等。

---

### （四）第三章：数组、指针和字符串 – 8 学时  
*（4 课堂讲授学时 + 4 实践学时）*  
**主要内容**:  
- 数组及对象数组  
- 指针及其与数组的关系  
- 动态内存分配  
- 深拷贝与浅拷贝  
- C-字符串与 string 型字符串的使用  
**教学要求**:  
- 熟练使用数组、指针与动态内存分配，掌握深拷贝与浅拷贝以及 string 字符串操作。  
**实践活动**:  
- 编写包含指针的学生类，进行信息加密、字符串统计、查找与替换等操作。

---

### （五）第四章：类的继承 – 8 学时  
*（4 课堂讲授学时 + 4 实践学时）*  
**主要内容**:  
- 继承和派生的基本概念  
- C++ 中继承的实现（private/protected/public 继承）  
- 单继承与多继承  
- 派生类构造与析构、替换原则、虚基类  
**教学要求**:  
- 掌握继承和派生的设计方法，能够熟练使用继承来设计类和编写程序。  
**实践活动**:  
- 编写时钟-闹钟类、点-圆类、人-学生类等实例。

---

### （六）第五章：多态性 – 虚函数和运算符重载 – 8 学时  
*（4 课堂讲授学时 + 4 实践学时）*  
**主要内容**:  
- 多态性概念及分类（动态绑定与静态绑定）  
- 虚函数的概念、实现及使用  
- 运算符重载（单目、双目及友元重载）  
**教学要求**:  
- 理解虚函数机制，熟练设计与使用虚函数及运算符重载。  
**实践活动**:  
- 编写几何形状类族、四则运算类族、复数类、人民币类、集合类、十六进制整数类、大整数类等。

---

### （七）第六章：模板与群体数据 – 4 学时  
*（2 课堂讲授学时 + 2 实践学时）*  
**主要内容**:  
- 泛型程序设计与模板的基本概念  
- 函数模板与类模板  
- 群体数据的组织  
**教学要求**:  
- 掌握函数模板和类模板的设计与使用。  
**实践活动**:  
- 实现插入排序函数模板、栈模板等。

---

### （八）第七章：泛型程序设计与 C++ 标准模板库（STL） – 4 学时  
*（2 课堂讲授学时 + 2 实践学时）*  
**主要内容**:  
- 泛型程序设计的基本概念与术语  
- STL 容器、迭代器、标准算法及函数对象  
**教学要求**:  
- 理解 STL，熟悉 vector、list、map、set、stack 等的使用。  
**实践活动**:  
- 编写词频统计、地理词汇英汉词典等程序。

---

### （九）第八章：流类库与输入输出 – 4 学时  
*（2 课堂讲授学时 + 2 实践学时）*  
**主要内容**:  
- I/O 流的基本概念、流类库结构  
- 标准输入输出、文件 I/O、字符串 I/O、宽字符输入输出  
**教学要求**:  
- 掌握流的概念及基本的输入输出操作。  
**实践活动**:  
- 处理中文字符文本文件、字符串流文件处理等。

---

### （十）第九章：异常处理 – 4 学时  
*（2 课堂讲授学时 + 2 实践学时）*  
**主要内容**:  
- 异常概念、异常处理机制在 C++ 中的实现  
- 异常的工作过程、异常接口声明与 C++ 标准异常  
**教学要求**:  
- 掌握异常机制，并能在程序中处理简单异常。  
**实践活动**:  
- 为动态数组类增加异常处理。

---

### （十一）第十章：Qt 程序设计介绍 – 4 学时  
*（2 课堂讲授学时 + 2 实践学时）*  
**主要内容**:  
- 介绍 Qt 框架及其开发基础  
- 界面设计组件、主框架窗体、对话框设计、事件系统、模型/视图结构、图形绘制等  
**教学要求**:  
- 掌握 Qt 开发入门，能设计简单的 GUI 应用程序。  
**实践活动**:  
- 设计一个简单的棋类游戏。

---

## 五、教材与学习资源

**主讲教材**:  
- 《C++语言程序设计（第5版）》，郑莉、董渊编著，清华大学出版社  
  - 出版日期：2020.11.01  
- **英文参考**:  
  - *C++ Primer* by Stanley B. Lippman, Josee Lajoie, Barbara E. Moo, 王刚等译，电子工业出版社, 2013  
  - *The C++ Programming Language* by Bjarne Stroustrup, 裘宗燕译，机械工业出版社, 2010  
  - *Effective C++* by Scott Meyers, 侯捷译，电子工业出版社, 2011  
  - 参考网站： [cppreference.com](http://en.cppreference.com), [cplusplus.com](http://www.cplusplus.com/reference/)

**注**: 感觉本课程主要依赖老师的 PPT 进行讲解，教材仅起辅助作用。

---

## 六、教学策略与方法建议

- **精讲多练**:  
  - 课堂讲解重点内容，辅以大量作业练习。  
- **作业点评与答疑**:  
  - 及时点评作业，针对性答疑，帮助同学们及时解决问题。  
- **上机实践**:  
  - 鼓励大家多上机实操，通过代码实现加深对知识的理解。

---

## 七、考核方式

- **成绩构成**:  
  - 平时成绩: 50%  
    - 包括：每章作业上机35分 + 上机思考题5分 + 课堂出勤5分 + 课堂作业5分  
  - 期末考试: 50%  
- **考试内容**:  
  - 期末考试题量较大，包含填空题、程序综合题（程序填空）、程序设计题  
  - 填空题侧重概念；综合题考查关键语法（如静态数据成员、模板定义）；程序设计题要求在给定情境下设计类、函数，关键在于知识的灵活运用和与题意的契合。

---

## 八、课程学习建议

- **知识点零散，需反复记忆**:  
  - C++ 建立在 C 语言基础上，理解 C++ 的各种特点及其使用至关重要。  
- **预习与复习**:  
  - 上课前务必预习，课后及时消化讲义内容；  
  - 利用老师的 PPT 结合教材，掌握重点知识；  
  - 建议不要堆积复习，应持续积累，避免期末临时抱佛脚。
- **多做练习**:  
  - 多做上机实操，熟悉各知识点在代码中的实际实现；  
  - 针对考试，特别是考研题型进行针对性训练，提升信心。
- **注重灵活运用**:  
  - 程序设计题要求变量名称、类与函数设计与题目保持一致，要注重实际应用与知识迁移。

---

## 九、成绩录入

- **录入速度**:  
  - 比较慢

---

# Object-oriented Method and Technology – Course Overview & Study Suggestions

**Course Code**: SAI11001 (BQ10214401 optional)  
**Course Type**: Required Core Course (Professional Basic and Core Course); Fundamental Discipline  
**Credits**: 2  
**Applicable Majors**: Computer Science, Artificial Intelligence  
**Language of Instruction**: Chinese  
**Total Hours**: 64  
**Semester Offered**: Spring  
**Times Offered**: 20  
**Recommended Enrollment**: 60 (with TA support)

**Instructor**: Shenjiali (Lecturer)  
**Contact**:  
- Email: sjl@bnu.edu.cn  
- Office Phone: (if available)

**Prerequisite**: Basics of Programming

---

## 1. Course Introduction

*Object-oriented Method and Technology* focuses on the fundamentals of object-oriented programming using C++.  
C++—developed by Bjarne Stroustrup in 1983 at Bell Labs—is an extension of C that incorporates object-oriented programming and supports generic programming through templates. It enables highly reusable code and algorithms by providing the Standard Template Library (STL) with containers, iterators, and algorithms.  
Overall, C++ blends procedural and object-oriented paradigms, offering efficiency, flexibility, and reusability. This course is designed to equip you with the basic knowledge and application skills of C++ to support further studies in computer programming, data structures, algorithm analysis, computer vision, and graphics.

---

## 2. Course Objectives

### Educational Objectives
- **Theory and Practice**:  
  Learn the fundamentals of C++ programming, including object-oriented concepts, class design, data encapsulation, inheritance, polymorphism, templates, and the STL.
- **Application**:  
  Apply these principles in program design and software development, preparing for advanced courses in computer science and AI.

### Holistic Goals
- Foster diligence, perseverance, and the courage to overcome challenges.  
- Stimulate exploration of new knowledge and technologies.  
- Enhance teamwork and collaborative problem-solving skills.

---

## 3. Course Content and Contact Hours

### (1) Introduction – 4 Hours  
*(2 hours lecture + 2 hours practice)*  
- **Topics**:  
  - Concepts of OOP vs. procedural programming  
  - History and advantages of OOP, C++ enhancements over C, generic programming  
  - Overview of C++ program structure, I/O, and development tools  
- **Teaching Requirements**:  
  - Master the features and advantages of object-oriented programming and the C++ development process.  
- **Practical Activities**:  
  - Install C++ development tools, write “Hello World!”, and implement basic I/O.

### (2) Chapter 1: Classes and Objects – 8 Hours  
*(4 hours lecture + 4 hours practice)*  
- **Topics**:  
  - Concepts of classes, design, access specifiers, data members, member functions, encapsulation, constructors, destructors, and the 'this' pointer.  
- **Teaching Requirements**:  
  - Master the design and implementation of classes, including constructors and destructors.  
- **Practical Activities**:  
  - Develop sample programs such as a rectangle class, a student class, and a birthday-enabled student class.

### (3) Chapter 2: Data Sharing and Protection – 8 Hours  
*(4 hours lecture + 4 hours practice)*  
- **Topics**:  
  - Scope and lifetime, static members, friend functions, const members and const objects.  
- **Teaching Requirements**:  
  - Master the design and usage of static members, friend functions, and const members for data sharing and protection.  
- **Practical Activities**:  
  - Implement examples like auto-numbering for students, banking classes, and use of const in function parameters.

### (4) Chapter 3: Arrays, Pointers, and Strings – 8 Hours  
*(4 hours lecture + 4 hours practice)*  
- **Topics**:  
  - Arrays, pointer arrays, dynamic memory allocation, deep vs. shallow copy, C-style and string class usage.  
- **Teaching Requirements**:  
  - Effectively use arrays, pointers, dynamic memory, and string manipulation in C++.  
- **Practical Activities**:  
  - Design a student class with pointers, perform encryption, and implement string search/replace functions.

### (5) Chapter 4: Inheritance – 8 Hours  
*(4 hours lecture + 4 hours practice)*  
- **Topics**:  
  - Concepts of inheritance and derivation, implementation in C++ (private/protected/public inheritance), single and multiple inheritance, constructors/destructors in derived classes, virtual base classes.  
- **Teaching Requirements**:  
  - Master design methods using inheritance and derivation to develop class hierarchies.  
- **Practical Activities**:  
  - Develop examples such as a clock-alarm class, point-circle class, and a human-student class.

### (6) Chapter 5: Polymorphism – Virtual Functions & Operator Overloading – 8 Hours  
*(4 hours lecture + 4 hours practice)*  
- **Topics**:  
  - Concepts of polymorphism (dynamic vs. static binding), virtual functions and their implementation, operator overloading (unary, binary, as friend or member functions).  
- **Teaching Requirements**:  
  - Understand and design virtual functions and operator overloading to build flexible applications.  
- **Practical Activities**:  
  - Develop examples such as a geometric shape hierarchy, arithmetic classes, complex number class, currency class, and large integer class.

### (7) Chapter 6: Templates and Aggregate Data – 4 Hours  
*(2 hours lecture + 2 hours practice)*  
- **Topics**:  
  - Generic programming concepts, function and class templates, template deduction, and aggregate data organization.  
- **Teaching Requirements**:  
  - Understand generic programming and master the design and use of function and class templates.  
- **Practical Activities**:  
  - Implement an insertion sort function template and a stack template.

### (8) Chapter 7: Generic Programming & STL – 4 Hours  
*(2 hours lecture + 2 hours practice)*  
- **Topics**:  
  - Concepts and terminology of generic programming, STL containers, iterators, standard algorithms, and function objects.  
- **Teaching Requirements**:  
  - Understand and utilize STL components such as vector, list, map, set, and stack.  
- **Practical Activities**:  
  - Create programs for word frequency statistics and a bilingual geographic dictionary.

### (9) Chapter 8: Stream Libraries & I/O – 4 Hours  
*(2 hours lecture + 2 hours practice)*  
- **Topics**:  
  - Concepts of I/O streams, structure of stream libraries, standard I/O, file I/O, string streams, and wide character strings.  
- **Teaching Requirements**:  
  - Master the basic I/O operations and file handling in C++.  
- **Practical Activities**:  
  - Design programs for processing text files and string stream operations.

### (10) Chapter 9: Exception Handling – 4 Hours  
*(2 hours lecture + 2 hours practice)*  
- **Topics**:  
  - Exception concepts, handling mechanisms in C++, exception processing, declaration of exception interfaces, and standard C++ exceptions.  
- **Teaching Requirements**:  
  - Master the use of exceptions and handling of simple errors in programs.  
- **Practical Activities**:  
  - Enhance a dynamic array class with exception handling.

### (11) Chapter 10: Introduction to Qt Programming – 4 Hours  
*(2 hours lecture + 2 hours practice)*  
- **Topics**:  
  - Overview of the Qt framework, basic Qt development, GUI design components, main window, dialog design, event system, model/view architecture, and graphics drawing.  
- **Teaching Requirements**:  
  - Learn the basics of Qt and develop simple GUI applications.  
- **Practical Activities**:  
  - Develop a chess game or similar application.

---

## 4. Textbooks and Learning Resources

**Main Textbook**:  
- *C++语言程序设计（第5版）*, 郑莉、董渊编著, 清华大学出版社 (2020.11.01)  
- **Reference Books**:  
  - *C++ Primer* by Stanley B. Lippman et al., 电子工业出版社, 2013  
  - *The C++ Programming Language* by Bjarne Stroustrup, 裘宗燕译, 机械工业出版社, 2010  
  - *Effective C++* by Scott Meyers, 侯捷译, 电子工业出版社, 2011  
  - Online: [cppreference.com](http://en.cppreference.com), [cplusplus.com](http://www.cplusplus.com/reference/)

**Additional Resources**:  
- Both Chinese and English electronic versions of textbooks and reference answer keys are available.
- Recommended online video lectures and animations for further visualization.

---

## 5. Teaching Strategies and Method Suggestions

- **Emphasis**:  
  - Focus on key concepts through detailed lecture explanations and ample practice problems.
- **Timely Homework Feedback**:  
  - Regular assignment review and prompt Q&A sessions to address difficulties.
- **Hands-On Practice**:  
  - Encourage extensive on-computer programming to consolidate theory with practice.

---

## 6. Assessment Methods

- **Grading Breakdown**:  
  - **Overall Score** = 50% (Coursework) + 50% (Final Exam)
  - **Coursework**:  
    - Based on per-chapter assignments including on-machine work (35 points per chapter) + on-machine thinking questions (5 points) + class attendance (5 points) + classroom assignments (5 points).
  - **Final Exam**: 50 points  
    - Exam includes fill-in-the-blank (focusing on conceptual content), comprehensive programming questions (code fill-in), and program design problems.
    - Programming design questions require supplementing given code with appropriate classes/functions; key in flexible application and consistency with problem statements.

---

## 7. Study Suggestions

- **Pre-study & Review**:  
  - Pre-read before class and consolidate lecture content immediately after; the instructor's pace can be fast, so prior preparation is crucial.
- **Emphasize Fundamentals**:  
  - The textbook contains detailed basic knowledge; pay close attention and review repeatedly.
- **Practice Extensively**:  
  - Complete all assigned exercises and, in addition, practice graduate-level exam questions to boost confidence.
- **Focus**:  
  - Strive to understand the underlying principles rather than rote memorization.
- **Utilize Provided Materials**:  
  - Make full use of the textbook (both Chinese and English versions), reference answer keys, and online resources.

---

## 8. Grade Entry

- **Grade Posting Speed**:  
  - Compared to other courses, grade entry is relatively slow.


