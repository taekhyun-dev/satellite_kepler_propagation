# test_init.py
import orekit_jpype
from jpype import getDefaultJVMPath, isJVMStarted, JPackage

# 1) JVM 시작
orekit_jpype.initVM(jvmpath=getDefaultJVMPath())

# 2) JVM이 제대로 올라갔는지 확인
print("JVM started?", isJVMStarted())  # True여야 정상

# 3) org.orekit 네임스페이스 접근
org = JPackage('org')
orekit_pkg = org.orekit

# 4) Orekit 버전 정보 가져오기
#    버전 클래스 이름은 Orekit 버전에 따라 다를 수 있으니 두 가지를 시도
try:
    version = orekit_pkg.Version.getVersion()
except AttributeError:
    version = orekit_pkg.OrekitConfig.getVersion()

print("Orekit version:", version)
