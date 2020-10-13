# typed: false
# frozen_string_literal: true

require "software_spec"

describe SoftwareSpec do
  subject(:spec) { described_class.new }

  describe "#uses_from_macos" do
    before do
      allow(OS).to receive(:mac?).and_return(true)
      allow(OS::Mac).to receive(:version).and_return(OS::Mac::Version.from_symbol(:sierra))
    end

    it "adds a macOS dependency if the OS version meets requirements" do
      spec.uses_from_macos("foo", since: :el_capitan)

      expect(spec.deps).to be_empty
      expect(spec.uses_from_macos_elements.first).to eq("foo")
    end

    it "doesn't add a macOS dependency if the OS version doesn't meet requirements" do
      spec.uses_from_macos("foo", since: :high_sierra)

      expect(spec.deps.first.name).to eq("foo")
      expect(spec.uses_from_macos_elements).to be_empty
    end

    it "works with tags" do
      spec.uses_from_macos("foo" => :build, :since => :high_sierra)

      dep = spec.deps.first

      expect(dep.name).to eq("foo")
      expect(dep.tags).to include(:build)
    end

    it "doesn't add a dependency if no OS version is specified" do
      spec.uses_from_macos("foo")
      spec.uses_from_macos("bar" => :build)

      expect(spec.deps).to be_empty
    end

    it "raises an error if passing invalid OS versions" do
      expect {
        spec.uses_from_macos("foo", since: :bar)
      }.to raise_error(MacOSVersionError, "unknown or unsupported macOS version: :bar")
    end
  end
end
