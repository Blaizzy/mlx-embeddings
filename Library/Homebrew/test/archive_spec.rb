# typed: false
# frozen_string_literal: true

require "archive"

describe Archive, :needs_network do
  subject(:archive) { described_class.new(item: "homebrew") }

  describe "::remote_checksum" do
    it "detects a published file" do
      hash = archive.remote_md5(directory: ".", remote_file: "cmake-3.1.2.yosemite.bottle.tar.gz")
      expect(hash).to eq("c6e525d472124670b0b635800488f438")
    end

    it "fails on a non-existent file" do
      hash = archive.remote_md5(directory: "bottles", remote_file: "my-fake-bottle-1.0.snow_hyena.tar.gz")
      expect(hash).to be nil
    end
  end
end
