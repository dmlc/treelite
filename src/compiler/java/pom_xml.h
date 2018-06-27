/*!
 * Copyright (c) 2017 by Contributors
 * \file pom_xml.h
 * \author Philip Cho
 * \brief template for pom.xml in generated Java code
 */
#ifndef TREELITE_COMPILER_JAVA_POM_XML_H_
#define TREELITE_COMPILER_JAVA_POM_XML_H_

namespace treelite {
namespace compiler {
namespace java {

const char* pom_xml =
"<project xmlns=\"http://maven.apache.org/POM/4.0.0\" "
"xmlns:xsi=\"http://www.w3.org/2001/XMLSchema-instance\"\n"
"xsi:schemaLocation=\"http://maven.apache.org/POM/4.0.0 "
"http://maven.apache.org/xsd/maven-4.0.0.xsd\">\n"
"<modelVersion>4.0.0</modelVersion>\n"
"\n"
"<build>\n"
"<plugins>\n"
"<plugin>\n"
"<groupId>org.apache.maven.plugins</groupId>\n"
"<artifactId>maven-shade-plugin</artifactId>\n"
"<version>1.6</version>\n"
"<executions>\n"
"<execution>\n"
"<phase>package</phase>\n"
"<goals>\n"
"<goal>shade</goal>\n"
"</goals>\n"
"</execution>\n"
"</executions>\n"
"</plugin>\n"
"</plugins>\n"
"</build>\n"
"\n"
"<groupId>treelite.predictor</groupId>\n"
"<artifactId>model</artifactId>\n"
"<version>1.0-SNAPSHOT</version>\n"
"<packaging>jar</packaging>\n"
"\n"
"<name>Maven Quick Start Archetype</name>\n"
"<url>http://maven.apache.org</url>\n"
"\n"
"<dependencies>\n"
"<dependency>\n"
"<groupId>org.javolution</groupId>\n"
"<artifactId>javolution-core-java</artifactId>\n"
"<version>6.0.0</version>\n"
"</dependency> \n"
"</dependencies>\n"
"</project>\n";

}  // namespace java
}  // namespace compiler
}  // namespace treelite
#endif  // TREELITE_COMPILER_JAVA_POM_XML_H_
