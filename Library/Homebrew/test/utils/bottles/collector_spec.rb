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
      subject[:yosemite] = "foo"
      expect(subject.fetch_checksum_for(:yosemite)).to eq(["foo", :yosemite])
      expect(subject.fetch_checksum_for(:mavericks)).to be nil
    end
  end
end
