# frozen_string_literal: true

require "software_spec"

describe SoftwareSpec do
  subject(:spec) { described_class.new }

  describe "#uses_from_macos" do
    before do
      sierra_os_version = OS::Mac::Version.from_symbol(:sierra)

      allow(OS).to receive(:mac?).and_return(true)
      allow(OS::Mac).to receive(:version).and_return(OS::Mac::Version.new(sierra_os_version))
    end

    it "allows specifying macOS dependencies before a certain version" do
      spec.uses_from_macos("foo", before: :high_sierra)

      expect(spec.deps).to be_empty
      expect(spec.uses_from_macos_elements.first).to eq("foo")
    end

    it "allows specifying macOS dependencies after a certain version" do
      spec.uses_from_macos("foo", after: :el_capitan)

      expect(spec.deps).to be_empty
      expect(spec.uses_from_macos_elements.first).to eq("foo")
    end

    it "doesn't add a macOS dependency if the OS version doesn't meet requirements" do
      spec.uses_from_macos("foo", after: :high_sierra)
      spec.uses_from_macos("bar", before: :el_capitan)

      expect(spec.deps.first.name).to eq("foo")
      expect(spec.uses_from_macos_elements).to be_empty
    end

    it "works with tags" do
      spec.uses_from_macos("foo" => :head, :after => :high_sierra)

      dep = spec.deps.first

      expect(dep.name).to eq("foo")
      expect(dep.tags).to include(:head)
    end

    it "doesn't add a dependency if no OS version is specified" do
      spec.uses_from_macos("foo")
      spec.uses_from_macos("bar" => :head)

      expect(spec.deps).to be_empty
    end

    it "respects OS version requirements with tags" do
      spec.uses_from_macos("foo" => :head, :before => :mojave)

      expect(spec.deps).to be_empty
    end

    it "raises an error if passing invalid OS versions" do
      expect {
        spec.uses_from_macos("foo", after: "bar", before: :mojave)
      }.to raise_error(ArgumentError, 'unknown version "bar"')
    end
  end
end
