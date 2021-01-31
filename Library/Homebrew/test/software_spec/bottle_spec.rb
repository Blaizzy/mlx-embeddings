# typed: false
# frozen_string_literal: true

require "software_spec"

describe BottleSpecification do
  subject(:bottle_spec) { described_class.new }

  describe "#sha256" do
    it "works without cellar" do
      checksums = {
        snow_leopard_32: "deadbeef" * 8,
        snow_leopard:    "faceb00c" * 8,
        lion:            "baadf00d" * 8,
        mountain_lion:   "8badf00d" * 8,
      }

      checksums.each_pair do |cat, digest|
        bottle_spec.sha256(digest => cat)
        checksum, = bottle_spec.checksum_for(cat)
        expect(Checksum.new(digest)).to eq(checksum)
      end
    end

    it "works with cellar" do
      checksums = [
        { cellar: :any_skip_relocation, tag: :snow_leopard_32, digest: "deadbeef" * 8 },
        { cellar: :any, tag: :snow_leopard, digest: "faceb00c" * 8 },
        { cellar: "/usr/local/Cellar", tag: :lion, digest: "baadf00d" * 8 },
        { cellar: Homebrew::DEFAULT_CELLAR, tag: :mountain_lion, digest: "8badf00d" * 8 },
      ]

      checksums.each do |checksum|
        bottle_spec.sha256(checksum[:tag] => checksum[:digest], cellar: checksum[:cellar])
        digest, tag, cellar = bottle_spec.checksum_for(checksum[:tag])
        expect(Checksum.new(checksum[:digest])).to eq(digest)
        expect(checksum[:tag]).to eq(tag)
        checksum[:cellar] ||= Homebrew::DEFAULT_CELLAR
        expect(checksum[:cellar]).to eq(cellar)
      end
    end
  end

  %w[root_url prefix cellar rebuild].each do |method|
    specify "##{method}" do
      object = Object.new
      bottle_spec.public_send(method, object)
      expect(bottle_spec.public_send(method)).to eq(object)
    end
  end
end
