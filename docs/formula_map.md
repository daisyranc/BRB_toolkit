# 公式与实现映射

R06/R07/M25链接见[使用文档](../README.zh-CN.md)。实现位于 brb/rimer 和 brb/maker。

|来源|公式/假设|实现|独立验证|
|---|---|---|---|
|R06 §IV|(16a,b)数值参考插值；显式信念输入|rimer/transform.py|端点、内部插值、缺失输入|
|R06 §III|(6a)幂积AND；(6b)加权product-sum OR；(7)激活|rimer/inference.py matching_and_activation|手算权重与参与掩码|
|R06 §III|(8)参与前件平均完整度更新beta|同上、model.py|不完整规则/输入、X6/X9 Unknown|
|R06 §III|(9–11)ER质量递归，区分输入未知与规则权重折扣|inference.py er_aggregate|单规则不完整、相反完整规则手算|
|R06 §III|(12a–c)效用上下界|model.py、results.py|未知输出必须明确决策|
|R06 §IV、V|多层信念直接传播|hierarchy.py、graphite.py|36规则中间节点、显式映射与环检测|
|R07 §III|(12)数值MSE；(15,16,19)逐后件与混合minimax|objectives.py、trainer.py|数值/主观/混合、冻结/专家约束|
|R07 §V|TableII数据、TableIV Expert1初始化|graphite_training.json|39规则、12个监督、完整规则等式、有序效用|
|M25 §2–3|归一化状态条件似然；非空集合焦元|evidence.py、state_space.py|Tables8–10、计数不是posterior|
|工程输入适配|原始DataFrame按可用样本获取单变量/前缀计数；显式平滑|acquisition.py|缺失模式、显式离散化、零支持与未知状态|
|M25 §3|(21,23)projection=1/(w·p+1-r)，m=w p|evidence.py Evidence|不可信残余单独存储|
|M25 §4|(49)alpha_AB=joint_AB/(p_A p_B)|combination.py dependency_indices|Table11首项1.7018|
|M25 §4|(51,54)联合权重指数与projection指数|_dependent_products|显式全1系数退化独立|
|M25 Theorem1|(59–61)bounded sum + 指数加权的intersection sum|combine_pair|Tables16–18完整数值与trustworthiness|
|M25 Corollaries1.1/1.3|(62,66)互斥与微分极限|combine_pair exclusive/derivative_rates|不可靠度增长率(1,2)手算|
|M25 Corollaries1.4/1.5|(67)独立；状态常量权重ER|combine_pair independent|集合交集、Bayes单一状态乘积|
|M25 Corollary1.6|(68)增广Dempster，tau及delta参数|combine_pair augment|普通Dempster与完全冲突微分极限|
|M25 Lemma2.1|(70–72)前缀ERPS质量递归，前缀获取量用于指数|extend_result、model.py|三变量前缀表、独立递归手算|
|M25 §5|(86–90)计数/总数 × 1/2平方偏差|objectives.py paper_objectives|独立观测项算式|
|M25 §5|(91–95)计数加权ESD与可靠度参考偏差的最大值|同上|参考参数冻结，权重偏差手算|
|M25 §5|(97–100)两阶段优先级minimax|trainer.py|阶段一失败终止、阶段二失败回滚、主目标上界|
|M25 §6|21参数诊断，单独训练组与联合训练组，独立prior|diagnosis.py、diagnosis.json|6300计数、Table18无prior、训练含prior|

OR的展开为 `1-product(1-normalized_delta*matching)`，与(6b)递归等价。无激活的完全未知返回、区间外端点截断、未见输入报错、资源上限和SLSQP内部离约束面试探，是明示的实现策略，不当作论文新结论。

M25的零边缘支撑但正联合支撑/无定义的权重指数，默认报错；不能靠平滑掩盖缺少联合信息。独立与互斥由使用者明确指定。只验证特例满足前提的退化，不声称任意模型自动等同Bayes或Dempster。
