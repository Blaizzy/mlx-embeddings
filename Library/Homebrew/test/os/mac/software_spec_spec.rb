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

    it "doesn't adds a dependency if it doesn't meet OS version requirements" do
      spec.uses_from_macos("foo", after: :high_sierra)
      spec.uses_from_macos("bar", before: :el_capitan)

      expect(spec.deps).to be_empty
    end

    it "allows specifying dependencies after certain version" do
      spec.uses_from_macos("foo", after: :el_capitan)

      expect(spec.deps.first.name).to eq("foo")
    end

    it "works with tags" do
      spec.uses_from_macos("foo" => :head, :after => :el_capitan)

      dep = spec.deps.first

      expect(dep.name).to eq("foo")
      expect(dep.tags).to include(:head)
    end

    it "allows specifying dependencies before certain version" do
      spec.uses_from_macos("foo", before: :high_sierra)

      expect(spec.deps.first.name).to eq("foo")
    end

    it "raises an error if passing invalid OS versions" do
      expect {
        spec.uses_from_macos("foo", after: "bar", before: :mojave)
      }.to raise_error(ArgumentError, 'unknown version "bar"')
    end
  end
end
