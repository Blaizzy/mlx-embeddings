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

    it "doesn't add a dependency" do
      spec.uses_from_macos("foo")
      spec.uses_from_macos("bar" => :build)

      expect(spec.deps).to be_empty
    end
  end
end
