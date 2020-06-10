# frozen_string_literal: true

require "utils/bottles"

describe Utils::Bottles::Collector do
  describe "#fetch_checksum_for" do
    it "returns passed tags" do
      subject[:mojave] = "foo"
      subject[:catalina] = "bar"
      expect(subject.fetch_checksum_for(:catalina)).to eq(["bar", :catalina])
    end

    it "returns nil if empty" do
      expect(subject.fetch_checksum_for(:foo)).to be nil
    end

    it "returns nil when there is no match" do
      subject[:catalina] = "foo"
      expect(subject.fetch_checksum_for(:foo)).to be nil
    end

    it "uses older tags when needed", :needs_macos do
      subject[:mojave] = "foo"
      expect(subject.send(:find_matching_tag, :mojave)).to eq(:mojave)
      expect(subject.send(:find_matching_tag, :catalina)).to eq(:mojave)
    end

    it "does not use older tags when requested not to", :needs_macos do
      allow(Homebrew::EnvConfig).to receive(:developer?).and_return(true)
      allow(Homebrew::EnvConfig).to receive(:skip_or_later_bottles?).and_return(true)
      allow(OS::Mac).to receive(:prerelease?).and_return(true)
      subject[:mojave] = "foo"
      expect(subject.send(:find_matching_tag, :mojave)).to eq(:mojave)
      expect(subject.send(:find_matching_tag, :catalina)).to be_nil
    end

    it "ignores HOMEBREW_SKIP_OR_LATER_BOTTLES on release versions", :needs_macos do
      allow(Homebrew::EnvConfig).to receive(:skip_or_later_bottles?).and_return(true)
      allow(OS::Mac).to receive(:prerelease?).and_return(false)
      subject[:mojave] = "foo"
      expect(subject.send(:find_matching_tag, :mojave)).to eq(:mojave)
      expect(subject.send(:find_matching_tag, :catalina)).to eq(:mojave)
    end
  end
end
