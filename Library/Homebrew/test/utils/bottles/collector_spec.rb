# typed: false
# frozen_string_literal: true

require "utils/bottles"

describe Utils::Bottles::Collector do
  subject(:collector) { described_class.new }

  describe "#fetch_checksum_for" do
    it "returns passed tags" do
      collector[:mojave] = { checksum: Checksum.new("foo_checksum"), cellar: "foo_cellar" }
      collector[:catalina] = { checksum: Checksum.new("bar_checksum"), cellar: "bar_cellar" }
      expect(collector.fetch_checksum_for(:catalina)).to eq(["bar_checksum", :catalina, "bar_cellar"])
    end

    it "returns nil if empty" do
      expect(collector.fetch_checksum_for(:foo)).to be nil
    end

    it "returns nil when there is no match" do
      collector[:catalina] = "foo"
      expect(collector.fetch_checksum_for(:foo)).to be nil
    end

    it "uses older tags when needed", :needs_macos do
      collector[:mojave] = "foo"
      expect(collector.send(:find_matching_tag, :mojave)).to eq(:mojave)
      expect(collector.send(:find_matching_tag, :catalina)).to eq(:mojave)
    end

    it "does not use older tags when requested not to", :needs_macos do
      allow(Homebrew::EnvConfig).to receive(:developer?).and_return(true)
      allow(Homebrew::EnvConfig).to receive(:skip_or_later_bottles?).and_return(true)
      allow(OS::Mac).to receive(:prerelease?).and_return(true)
      collector[:mojave] = "foo"
      expect(collector.send(:find_matching_tag, :mojave)).to eq(:mojave)
      expect(collector.send(:find_matching_tag, :catalina)).to be_nil
    end

    it "ignores HOMEBREW_SKIP_OR_LATER_BOTTLES on release versions", :needs_macos do
      allow(Homebrew::EnvConfig).to receive(:skip_or_later_bottles?).and_return(true)
      allow(OS::Mac).to receive(:prerelease?).and_return(false)
      collector[:mojave] = "foo"
      expect(collector.send(:find_matching_tag, :mojave)).to eq(:mojave)
      expect(collector.send(:find_matching_tag, :catalina)).to eq(:mojave)
    end
  end
end
