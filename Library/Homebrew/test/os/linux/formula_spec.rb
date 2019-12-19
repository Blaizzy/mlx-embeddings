# frozen_string_literal: true

require "formula"

describe Formula do
  describe "#uses_from_macos" do
    before do
      allow(OS).to receive(:mac?).and_return(false)
    end

    it "acts like #depends_on" do
      f = formula "foo" do
        url "foo-1.0"

        uses_from_macos("foo")
      end

      expect(f.class.stable.deps.first.name).to eq("foo")
      expect(f.class.devel.deps.first.name).to eq("foo")
      expect(f.class.head.deps.first.name).to eq("foo")
    end
  end
end
