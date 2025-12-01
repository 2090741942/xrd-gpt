from pymatgen.io.vasp import Poscar
from pymatgen.analysis.structure_matcher import StructureMatcher
import math

# poscar格式的生成数据
poscar_str1 = "System\n1.0\n3.51 0.0 0.0\n0.0 3.51 0.0\n0.0 0.0 5.41\nMg Be Sn \n1 2 1 \ndirect\n0.5 0.5 0.0 Mg\n0.0 0.0 0.773 Be\n0.0 0.0 0.227 Be\n0.5 0.5 0.5 Sn\n,System\n1.0\n4.39521 0.0 -0.69613\n-0.7417 4.38157 -0.2329\n0.0 0.0 5.25\nMg Be Sn \n1 2 1 \ndirect\n0.633 0.632 0.835 Mg\n0.0 0.0 0.0 Be\n0.359 0.352 0.596 Be\n0.178 0.178 0.287 Sn\n"

# poscar_str2 = "System\n1.0\n4.29 0.0 0.0\n0.0 4.29 0.0\n0.0 0.0 9.75\nTm Cu Sb \n2 2 4 \ndirect\n0.75 0.75 0.751 Tm\n0.25 0.25 0.249 Tm\n0.25 0.75 0.5 Cu\n0.75 0.25 0.5 Cu\n0.25 0.75 0.0 Sb\n0.75 0.25 0.0 Sb\n0.75 0.75 0.334 Sb\n0.25 0.25 0.666 Sb\n,System\n1.0\n4.07915 0.0 -0.64607\n-0.74894 4.05897 -0.14413\n0.0 0.0 10.99\nTm Cu Sb \n1 1 2 \ndirect\n0.0 0.0 0.0 Tm\n0.5 0.5 0.0 Cu\n0.5 0.5 0.5 Sb\n0.0 0.0 0.5 Sb\n"

poscar_str2 = "System\n1.0\n3.51 0.0 0.0\n0.0 3.51 0.0\n0.0 0.0 5.41\nMg Be Sn \n1 2 1 \ndirect\n0.5 0.5 0.0 Mg\n0.0 0.0 0.773 Be\n0.0 0.0 0.227 Be\n0.5 0.5 0.5 Sn\n,System\n1.0\n4.39521 0.0 -0.69613\n-0.7417 4.38157 -0.2329\n0.0 0.0 5.25\nMg Be Sn \n1 2 1 \ndirect\n0.633 0.632 0.835 Mg\n0.0 0.0 0.0 Be\n0.359 0.352 0.596 Be\n0.178 0.178 0.287 Sn\n"


# 2. 转成 pymatgen 的 Structure
struct1 = Poscar.from_str(poscar_str1).structure
struct2 = Poscar.from_str(poscar_str2).structure

# 3. 建一个 StructureMatcher
sm = StructureMatcher(
    ltol=0.2,      # 晶格常数容差（相对）
    stol=0.3,      # 原子坐标的容差（分数坐标）
    angle_tol=5    # 角度容差（度）
)

# 4. 判断是否匹配
is_match = sm.fit(struct1, struct2)
print("是否匹配：", is_match)

# 量化“差多少”
if sm.get_rms_dist(struct1, struct2):
    rms, max_dist = sm.get_rms_dist(struct1, struct2)
    print("RMS 位移：", rms)
    print("最大原子位移：", max_dist)

    similarity1 = 1 / (1 + rms)   
    similarity2 = math.exp(-rms)
    print(f"反比例函数：{similarity1}， 指数衰减：{similarity2}")