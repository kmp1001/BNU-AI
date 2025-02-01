# 人工智能数学基础 课程信息与学习建议
**英文名称**：Mathematical Foundation to Artificial Intelligence

---

## 课程基本信息

- **课程代码**：SAI02003A  
- **课程性质**：专业必修课（专业基础课）  
- **学分数**：2  
- **适用专业**：人工智能  
- **授课语言**：中文授课  
- **学时数**：32  
- **开设学期**：春季  
- **已开设次数**：3  
- **建议最大选课人数**：60（配备助教）  

**授课教师**：张家才 教授  
**联系方式**：  
- Email：jiacai.zhang@bnu.edu.cn  
- 手机：13501088946  

**先修课程**：微积分、线性代数、概率统计、Python 程序设计

---

## 一、课程简介

本课程重新梳理作为人工智能领域数学基础的核心内容，包括微积分、线性代数、概率统计等基础知识。课程旨在激活并筑牢人工智能研究与应用中常见的数学方法，补充本科数学课程中未涉及的数值分析与优化理论内容，帮助大家建立数学与人工智能的内在联系，提升数学基础的理解与应用能力，同时掌握利用计算机编程工具（如 Python）解决实际问题的技能。

---

## 二、课程目标

### 教学目标
1. **向量、矩阵与张量**  
   - 深入理解标量、向量、矩阵及张量的概念和几何意义；  
   - 掌握向量与矩阵范数的定义及几何意义，熟悉单位矩阵、对称矩阵、对角矩阵、正交矩阵和正定矩阵等常见矩阵的特点；  
   - 熟练掌握矩阵的运算技巧，包括矩阵求逆、伪逆、高斯消去法等。

2. **矩阵分解与线性变换**  
   - 理解特征值、特征向量、矩阵正交分解、三角分解、奇异值分解等方法；  
   - 能够应用这些方法解决文本与图像表征、主成分分析、投影等实际问题。

3. **多变量微积分与最优化**  
   - 掌握函数极限、导数、偏导数和梯度的概念，理解泰勒公式、链式法则等；  
   - 理解梯度下降、随机梯度下降、牛顿法等优化算法，并掌握神经网络中后向传播算法的基本原理；  
   - 熟悉最优化理论、拉格朗日乘子法、对偶理论以及动态规划等内容。

4. **概率统计基础与应用**  
   - 准确理解概率、条件概率、联合概率及概率密度的定义；  
   - 熟练掌握贝叶斯公式、最大似然估计、最大后验估计和最大期望估计；  
   - 理解马尔可夫模型及隐马尔可夫模型的基本原理，并掌握前向、后向算法和维特比算法在自然语言处理等领域的应用。

5. **综合实践能力**  
   - 培养学生利用数学知识独立分析和解决实际问题的能力，掌握基于 Python 的数学计算与仿真工具；  
   - 强调数学与计算思维的结合，提升创新设计和工程实践能力。

### 育人目标
- 培养数学理论与动手能力协调发展的人工智能专业人才，树立正确的科学观、学术规范和人工智能伦理；  
- 引导学生关注数学的逻辑美和表达美，增强对前沿技术与实际问题解决方案的理解；  
- 培养学生的创新能力、实践能力以及团队协作和交流能力，激发家国情怀。

---

## 三、教学内容与学时分配

本课程总计 32 学时，内容涵盖数学基础在人工智能中的应用，主要模块包括：

### （一）总论（绪论、概论） – 2 学时
- **主要内容**：
  - 人工智能核心任务：建模  
  - 建模中的数学表示、约束与求解  
  - 结合“农夫过河”问题讲解建模、表示、约束和求解的全过程  
  - 课程教学内容、目标及考核方式解读
- **教学要求**：  
  通过具体案例让大家体会数学建模的美感，认识到数学不仅提供计算工具，更为问题描述与求解提供数据结构和算法基础。

### （二）线性代数：向量与矩阵的基本概念 – 4 学时
- **主要内容**：
  - 向量、矩阵和张量的基本概念及几何意义  
  - 从数据结构角度理解矩阵的应用（如物品推荐、文本相似度）  
  - 矩阵与向量的基本运算、行列式、主子式及矩阵秩的几何意义  
  - 演示 AutoEncoder 与 Word2Vec 代码，讲解向量和矩阵在实际问题中的应用
- **教学要求**：  
  帮助大家从更广义的角度理解矩阵和向量，并培养利用 Python 工具进行数学计算的能力。

### （三）线性代数：线性方程组 – 4 学时
- **主要内容**：
  - 齐次与非齐次线性方程组的矩阵表示及解的条件  
  - 利用高斯消去法（及其变形）求解线性方程组与矩阵求逆
- **教学要求**：  
  使大家从列向量线性组合的角度理解线性方程组，并加深对矩阵零空间与秩的理解。

### （四）线性代数：向量空间与变换 – 4 学时
- **主要内容**：
  - 向量空间、基、维度、基变换与过渡矩阵  
  - 向量的变换（旋转、缩放、剪切）及其矩阵表示  
  - 矩阵特征值、特征向量、正交变换及 Gram-Schmidt 过程  
  - 投影变换及矩阵的四大子空间
- **教学要求**：  
  将抽象的数学运算形象化，帮助大家建立起从感性到理性的数学认知。

### （五）线性代数：综合应用 – 2 学时
- **主要内容**：
  - 矩阵的各种分解（奇异值分解、特征分解等）、相似变换、对角化  
  - 二次型、正定与半正定矩阵  
  - 结合实例进行图片压缩、商品推荐、主成分分析及 PageRank 算法讲解
- **教学要求**：  
  通过经典应用深化对线性代数各知识点及变换几何意义的理解。

### （六）微积分：函数、极限与导数 – 2 学时
- **主要内容**：
  - 函数、极限、连续性与导数，常见函数求导公式  
  - 泰勒公式、麦克劳林展开，不定积分与定积分基本理论
- **教学要求**：  
  复习基本概念，厘清连续与可导的条件，并掌握利用极限思想进行公式推导的方法。

### （七）微积分：偏导数、梯度与梯度下降法 – 4 学时
- **主要内容**：
  - 偏导数、方向导数与多变量求导链式法则  
  - 梯度、极值求解与梯度下降法  
  - 综合应用：多项式回归分析与神经网络后向传播（BP）算法的推导
- **教学要求**：  
  理解梯度在函数最值求解中的作用，为人工神经网络中的误差反向传播打下基础。

### （八）微积分：最优化理论 – 4 学时
- **主要内容**：
  - 约束极值问题、线性规划、单纯形法、对偶理论  
  - 二次规划与拉格朗日方法，动态规划基本原理
- **教学要求**：  
  掌握最优化理论基础，理解约束条件在最优化问题中的作用及其转化技巧。

### （九）概率统计：概率论基础 – 2 学时
- **主要内容**：
  - 随机事件、条件与联合概率  
  - 连续与离散变量、概率密度与累积函数  
  - 典型概率分布（均匀、二项、高斯等）及中心极限定理  
  - 综合应用：基于主成分分析的人脸图像识别
- **教学要求**：  
  理解连续与离散概率的不同，掌握概率密度函数与定积分的关系。

### （十）概率统计：随机变量与参数估计 – 2 学时
- **主要内容**：
  - 贝叶斯公式与决策理论  
  - 数理统计基本概念（均值、方差、协方差）  
  - 最大似然、最大后验、最大期望估计  
- **教学要求**：  
  通过参数估计方法，理解概率统计在人工智能中的应用范式。

### （十一）概率统计：Markov 模型及其应用 – 2 学时
- **主要内容**：
  - 马尔可夫模型与隐马尔可夫模型的定义、状态空间及转移概率矩阵  
  - 前向后向算法、维特比算法  
  - 综合应用：基于隐马尔可夫模型的中文分词
- **教学要求**：  
  结合动态规划思路，理解隐马尔可夫模型在自然语言处理中的应用。

---

## 四、选用教材与学习资源

1. **《人工智能的数学基础》**  
   - 主编：唐宇迪、李琳、侯惠芳、王社伟  
   - 出版社：北京大学出版社  
   - ISBN：9787301314319  
   - 注意：该教材内容较差，不推荐购买。

2. **《Linear Algebra and Learning from Data》**  
   - 作者：Gilbert Strang  
   - 出版社：Wellesley College  
   - ISBN：9780692196380

---

## 五、教学策略与方法建议

1. **计算思维与数学思维的结合**  
   - 本课程为数学基础与计算实践完美结合的机会。课堂上鼓励同学们携带电脑，及时使用 Python 代码实现数学运算，通过实践加深理解。

2. **利用公开数据集与经典算法**  
   - 通过网络公开数据集和人工智能相关算法，将数学表达、约束建模和求解过程具体化，激发大家对数学的兴趣和对应用潜力的认识。

3. **互动式教学**  
   - 课堂以讲解为主，辅以讨论、实验展示和课后编程练习。老师注重实时反馈，帮助同学们及时纠正问题。

---

## 六、考核方式

- **综合成绩（100 分）**  
  = 平时成绩（50%） + 期末成绩（50%）

- **平时成绩（100 分）**  
  = 5 次课后编程实验总分（每次满分 20 分），评价标准包括：  
  - 实验报告质量  
  - 解决问题的独创性  
  - Python 代码可读性  
  - 数学知识的运用深度  

- **期末成绩（100 分）**  
  - 闭卷考试  
  - 试题难度不大，但张老师可能会考察一些偏题内容，例如：推导多元正态分布中**独立性**与**不相关性**（如何表达？）等价问题。

---

## 七、课程学习建议

- **听课与复习**  
  - 该门课讲授节奏较快，请大家务必跟上课程进度。  
  - 课前预习和课后复习一定要重视，重点复习 lecture slides，特别是关键公式和推导过程。

- **实验报告**  
  - 平时成绩主要取决于实验报告的质量，建议大家在报告中充分体现个人思路、学习过程和详细的数学公式，同时注意格式设计。

- **参考资料**  
  - 建议大家利用往年真题进行复习，并参考公开数据集与经典算法，增强理论与实践结合的能力。

- **提高动手能力**  
  - 利用 Python 编程实验加深对数学知识的理解，养成独立解决问题的习惯。

---

# Mathematical Foundation to Artificial Intelligence – Course Overview & Study Suggestions

**Course Code**: SAI02003A  
**Course Type**: Required Core Course  
**Credits**: 2  
**Applicable Major**: Artificial Intelligence  
**Language of Instruction**: Chinese  
**Total Hours**: 32  
**Semester Offered**: Spring  
**Maximum Enrollment**: 60 (with TA support)  

**Instructor**: Prof. Zhang Jiacai  
**Contact**:  
- Email: jiacai.zhang@bnu.edu.cn  
- Phone: 13501088946  

**Prerequisites**: Calculus, Linear Algebra, Probability & Statistics, Python Programming

---

## 1. Course Introduction

This course revisits the fundamental mathematical concepts crucial to the field of Artificial Intelligence, including Calculus, Linear Algebra, and Probability & Statistics. It aims to reinforce and activate common mathematical methods used in AI research and applications, supplementing topics such as numerical analysis and optimization theory that are not covered in standard undergraduate mathematics courses. Additionally, the course emphasizes the use of computer programming (Python) to solve mathematical problems in AI.

---

## 2. Course Objectives

### Educational Objectives
1. **Concepts of Scalars, Vectors, Matrices, and Tensors**  
   - Gain a deep understanding of these concepts and their geometric meanings; master norms and common matrix properties (identity, symmetric, diagonal, orthogonal, positive definite).
   - Learn various matrix operations, including inversion, pseudo-inverse, and Gaussian elimination.

2. **Matrix Decompositions & Linear Transformations**  
   - Understand eigenvalues, eigenvectors, orthogonal and triangular decompositions, singular value decomposition, and their applications in data representation (e.g., text and image processing).

3. **Multivariable Calculus & Optimization**  
   - Master limits, derivatives, partial derivatives, gradients, and the chain rule.
   - Understand optimization algorithms such as gradient descent, stochastic gradient descent, Newton's method, and apply them in neural network backpropagation.

4. **Probability & Statistics Foundations**  
   - Accurately grasp probability, conditional probability, joint probability, and probability density.
   - Learn Bayesian inference, maximum likelihood, maximum a posteriori, and expectation-maximization methods.
   - Understand Markov models and Hidden Markov Models (HMM), including forward-backward and Viterbi algorithms.

5. **Integrated Practical Skills**  
   - Develop the ability to independently analyze and solve problems using mathematics and Python-based simulation tools.
   - Emphasize the combination of mathematical thinking with computational methods to foster innovation and practical problem-solving.

### Holistic Goals
- Cultivate AI professionals with balanced theoretical and practical skills, proper scientific attitudes, academic ethics, and an appreciation for the logical beauty of mathematics.
- Enhance problem-solving and teamwork abilities, and inspire a sense of responsibility and national pride through comparisons of international research frontiers.

---

## 3. Course Content & Contact Hours

The course is 32 hours in total. Key modules include:

### (1) Introduction – 2 Hours
- **Topics**:  
  - Core AI tasks: modeling  
  - Mathematical representation, constraints, and solving in AI  
  - Case study: “Farmer Crossing the River” to illustrate the entire process  
  - Overview of course content, objectives, and assessment methods  
- **Objective**:  
  Appreciate how flexible application of mathematical knowledge is crucial to AI implementation.

### (2) Linear Algebra: Basic Concepts of Vectors and Matrices – 4 Hours  
- **Topics**:  
  - Understanding vectors, matrices, and tensors from a data structure perspective  
  - Applications in recommendation systems and text similarity  
  - Basic operations, determinants, and geometric meanings  
  - Demonstration with AutoEncoder and Word2Vec code  
- **Objective**:  
  Broaden your understanding of matrix operations and their powerful applications using Python.

### (3) Linear Algebra: Systems of Linear Equations – 4 Hours  
- **Topics**:  
  - Representation of homogeneous and non-homogeneous systems  
  - Conditions for unique, no, or infinite solutions  
  - Solving systems via Gaussian elimination and understanding the concept of the matrix’s null space  
- **Objective**:  
  Shift from viewing equations individually to understanding them as a system of vector combinations.

### (4) Linear Algebra: Vector Spaces & Transformations – 4 Hours  
- **Topics**:  
  - Basis, dimension, coordinate transformations, and transition matrices  
  - Geometric interpretations of rotation, scaling, and projection  
  - Eigenvalues, eigenvectors, and orthogonal transformations  
- **Objective**:  
  Visualize abstract algebraic operations in high-dimensional space.

### (5) Linear Algebra: Integrated Applications – 2 Hours  
- **Topics**:  
  - Various matrix decompositions (e.g., SVD, eigen decomposition)  
  - Diagonalization, quadratic forms, and applications in image compression, recommendation systems, PCA, and PageRank  
- **Objective**:  
  Deepen understanding through classic applications and real-world examples.

### (6) Calculus: Functions, Limits, and Derivatives – 2 Hours  
- **Topics**:  
  - Fundamentals of functions, limits, continuity, and derivatives  
  - Taylor and Maclaurin series, indefinite and definite integrals  
- **Objective**:  
  Review and consolidate key calculus concepts essential for advanced topics.

### (7) Calculus: Partial Derivatives, Gradients & Gradient Descent – 4 Hours  
- **Topics**:  
  - Partial derivatives, directional derivatives, and the chain rule  
  - Gradient, extreme values, and gradient descent  
  - Applications: polynomial regression and neural network backpropagation
- **Objective**:  
  Understand how gradients guide optimization and learn to derive neural network error gradients.

### (8) Calculus: Optimization Theory – 4 Hours  
- **Topics**:  
  - Constrained optimization, linear programming (Simplex Method), duality  
  - Quadratic programming and Lagrange multipliers, dynamic programming  
- **Objective**:  
  Master foundational optimization techniques for both unconstrained and constrained problems.

### (9) Probability & Statistics: Fundamentals – 2 Hours  
- **Topics**:  
  - Random events, conditional and joint probability  
  - Discrete vs. continuous variables, density functions, typical distributions  
  - Application: PCA-based face recognition
- **Objective**:  
  Understand the link between density functions and integration.

### (10) Probability & Statistics: Random Variables & Parameter Estimation – 2 Hours  
- **Topics**:  
  - Bayesian formula and decision theory  
  - Basic statistical concepts: mean, variance, covariance  
  - Maximum likelihood, MAP, and EM methods  
- **Objective**:  
  Grasp estimation techniques and their relevance to AI problem solving.

### (11) Probability & Statistics: Markov Models & Applications – 2 Hours  
- **Topics**:  
  - Definitions of Markov and Hidden Markov Models  
  - Forward-backward algorithm and Viterbi algorithm  
  - Application: Chinese word segmentation using HMM
- **Objective**:  
  Understand dynamic programming solutions in probabilistic models.

---

## 4. Selected Textbooks & Learning Resources

1. **"Mathematical Foundation to Artificial Intelligence"**  
   - Editor: Tang Yudi et al.  
   - Publisher: Peking University Press  
   - ISBN: 9787301314319  
   - **Note**: The content of this textbook is considered subpar; it is not recommended for purchase.

2. **"Linear Algebra and Learning from Data"**  
   - Author: Gilbert Strang  
   - Publisher: Wellesley College  
   - ISBN: 9780692196380

---

## 5. Teaching Strategies & Method Suggestions

1. **Integrating Computational & Mathematical Thinking**  
   - The course offers a great opportunity to combine mathematical theory with practical computation. Students are encouraged to bring their laptops and use Python to implement mathematical operations in real time.

2. **Leveraging Open Data & Classic Algorithms**  
   - Utilize publicly available datasets and classic AI algorithms to extract and derive mathematical models, fostering independent analytical and programming skills.

3. **Interactive Classroom Approach**  
   - The teaching method includes lectures, discussions, experimental demonstrations, and after-class programming exercises to ensure continuous feedback and understanding.

---

## 6. Assessment Methods

- **Overall Score (100 Points)**  
  = 50% Coursework + 50% Final Exam

- **Coursework (100 Points)**  
  - Based on 5 after-class programming experiments (each worth 20 points).  
  - Evaluation criteria include:
    - Quality of lab reports (emphasis on personal thought process, detailed mathematical derivations, and proper formatting)  
    - Originality in problem solving  
    - Readability of Python code  
    - Depth in the application of mathematical knowledge

- **Final Exam (100 Points)**  
  - Closed-book exam with relatively easy questions.  
  - Note: Some slightly unconventional topics may be tested (e.g., derivations related to the independence vs. uncorrelatedness of a multivariate normal distribution).

- **Grade Posting**:  
  - Grades are entered quickly.

---

## 7. Study Suggestions

- **Class Attendance & Review**  
  - The instructor’s pace is fast; ensure you follow along and review the lecture slides thoroughly.

- **Lab Reports**  
  - Since lab report quality is crucial to your coursework grade, make sure to document your thought process, include ample formulas, and design your report with clear formatting.

- **Practice with Past Exams**  
  - Utilize past exam papers to familiarize yourself with the test format and key topics.

- **Hands-On Practice**  
  - Reinforce your learning by completing Python programming exercises to apply mathematical concepts to real-world AI problems.



