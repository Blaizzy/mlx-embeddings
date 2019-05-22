# frozen_string_literal: true

require "formula"

describe Formula do
  describe "#uses_from_macos" do
    before do
      sierra_os_version = OS::Mac::Version.from_symbol(:sierra)

      allow(OS).to receive(:mac?).and_return(true)
      allow(OS::Mac).to receive(:version).and_return(OS::Mac::Version.new(sierra_os_version))
    end

    it "doesn't adds a dependency if it doesn't meet OS version requirements" do
      f = formula "foo" do
        url "foo-1.0"

        uses_from_macos("foo", after: :high_sierra)
        uses_from_macos("bar", before: :el_capitan)
      end

      expect(f.class.stable.deps).to be_empty
      expect(f.class.devel.deps).to be_empty
      expect(f.class.head.deps).to be_empty
    end

    it "allows specifying dependencies after certain version" do
      f = formula "foo" do
        url "foo-1.0"

        uses_from_macos("foo", after: :el_capitan)
      end

      expect(f.class.stable.deps.first.name).to eq("foo")
      expect(f.class.devel.deps.first.name).to eq("foo")
      expect(f.class.head.deps.first.name).to eq("foo")
    end

    it "allows specifying dependencies before certain version" do
      f = formula "foo" do
        url "foo-1.0"

        uses_from_macos("foo", before: :high_sierra)
      end

      expect(f.class.stable.deps.first.name).to eq("foo")
      expect(f.class.devel.deps.first.name).to eq("foo")
      expect(f.class.head.deps.first.name).to eq("foo")
    end

    it "raises an error if passing invalid OS versions" do
      expect {
        formula "foo" do
          url "foo-1.0"

          uses_from_macos("foo", after: "bar", before: :mojave)
        end
      }.to raise_error(ArgumentError, 'unknown version "bar"')
    end
  end
end
