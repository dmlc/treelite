const char* pom_xml_template =
R"TREELITETEMPLATE(
<project xmlns="http://maven.apache.org/POM/4.0.0" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
xsi:schemaLocation="http://maven.apache.org/POM/4.0.0 http://maven.apache.org/xsd/maven-4.0.0.xsd">
<modelVersion>4.0.0</modelVersion>

<build>
<plugins>
<plugin>
<groupId>org.apache.maven.plugins</groupId>
<artifactId>maven-shade-plugin</artifactId>
<version>1.6</version>
<executions>
<execution>
<phase>package</phase>
<goals>
<goal>shade</goal>
</goals>
</execution>
</executions>
</plugin>
</plugins>
</build>

<groupId>{java_package}</groupId>
<artifactId>model</artifactId>
<version>{java_package_version}</version>
<packaging>jar</packaging>

<name>Maven Quick Start Archetype</name>
<url>http://maven.apache.org</url>

<dependencies>
<dependency>
<groupId>org.javolution</groupId>
<artifactId>javolution-core-java</artifactId>
<version>6.0.0</version>
</dependency>
</dependencies>
</project>
)TREELITETEMPLATE";
