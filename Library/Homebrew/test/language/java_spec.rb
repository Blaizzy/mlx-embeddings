# typed: false
# frozen_string_literal: true

require "language/java"

describe Language::Java do
  let(:f) do
    formula("openjdk") do
      url "openjdk"
      version "15.0.1"
    end
  end

  before do
    allow(Formula).to receive(:[]).and_return(f)
    allow(f).to receive(:any_version_installed?).and_return(true)
    allow(f).to receive(:any_installed_version).and_return(f.version)
  end

  describe "::java_home" do
    it "returns valid JAVA_HOME if version is specified", :needs_macos do
      java_home = described_class.java_home("1.8+")
      expect(java_home).to eql(f.opt_libexec/"openjdk.jdk/Contents/Home")
    end

    it "returns valid JAVA_HOME if version is not specified", :needs_macos do
      java_home = described_class.java_home
      expect(java_home).to eql(f.opt_libexec/"openjdk.jdk/Contents/Home")
    end

    it "returns valid JAVA_HOME if version is specified", :needs_linux do
      java_home = described_class.java_home("1.8+")
      expect(java_home).to eql(f.opt_libexec)
    end

    it "returns valid JAVA_HOME if version is not specified", :needs_linux do
      java_home = described_class.java_home
      expect(java_home).to eql(f.opt_libexec)
    end
  end

  describe "::java_home_env" do
    it "returns java_home path if version specified" do
      java_home_env = described_class.java_home_env("1.8+")
      expect(java_home_env[:JAVA_HOME]).to include(f.opt_libexec.to_s)
    end

    it "returns java_home path if version is not specified" do
      java_home_env = described_class.java_home_env
      expect(java_home_env[:JAVA_HOME]).to include(f.opt_libexec.to_s)
    end
  end

  describe "::overridable_java_home_env" do
    it "returns java_home path if version specified" do
      overridable_java_home_env = described_class.overridable_java_home_env("1.8+")
      expect(overridable_java_home_env[:JAVA_HOME]).to include(f.opt_libexec.to_s)
    end

    it "returns java_home path if version is not specified" do
      overridable_java_home_env = described_class.overridable_java_home_env
      expect(overridable_java_home_env[:JAVA_HOME]).to include(f.opt_libexec.to_s)
    end
  end
end
