# frozen_string_literal: true

require "utils/bottles"

describe Utils::Bottles::Collector do
  describe "#fetch_checksum_for" do
    it "returns passed tags" do
      subject[:yosemite] = "foo"
      subject[:el_captain] = "bar"
      expect(subject.fetch_checksum_for(:el_captain)).to eq(["bar", :el_captain])
    end

    it "returns nil if empty" do
      expect(subject.fetch_checksum_for(:foo)).to be nil
    end

    it "returns nil when there is no match" do
      subject[:yosemite] = "foo"
      expect(subject.fetch_checksum_for(:foo)).to be nil
    end

    it "uses older tags when needed", :needs_macos do
      subject[:mavericks] = "foo"
      expect(subject.send(:find_matching_tag, :mavericks)).to eq(:mavericks)
      expect(subject.send(:find_matching_tag, :yosemite)).to eq(:mavericks)
    end

    it "does not use older tags when requested not to", :needs_macos do
      allow(ARGV).to receive(:skip_or_later_bottles?).and_return(true)
      allow(OS::Mac).to receive(:prerelease?).and_return(true)
      subject[:mavericks] = "foo"
      expect(subject.send(:find_matching_tag, :mavericks)).to eq(:mavericks)
      expect(subject.send(:find_matching_tag, :yosemite)).to be_nil
    end

    it "ignores HOMEBREW_SKIP_OR_LATER_BOTTLES on release versions", :needs_macos do
      allow(ARGV).to receive(:skip_or_later_bottles?).and_return(true)
      allow(OS::Mac).to receive(:prerelease?).and_return(false)
      subject[:mavericks] = "foo"
      expect(subject.send(:find_matching_tag, :mavericks)).to eq(:mavericks)
      expect(subject.send(:find_matching_tag, :yosemite)).to eq(:mavericks)
    end
  end
end
