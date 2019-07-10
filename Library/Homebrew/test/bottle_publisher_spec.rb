# frozen_string_literal: true

require "bottle_publisher"

describe BottlePublisher do
  subject(:bottle_publisher) {
    described_class.new(
      CoreTap.instance, ["#{CoreTap.instance.name}/hello.rb"], "homebrew", false, false
    )
  }

  let(:tap) { CoreTap.new }

  describe "publish_and_check_bottles" do
    it "fails if HOMEBREW_DISABLE_LOAD_FORMULA is set to 1" do
      ENV["HOMEBREW_DISABLE_LOAD_FORMULA"] = "1"
      expect { bottle_publisher.publish_and_check_bottles }
        .to raise_error("Need to load formulae to publish them!")
    end

    it "returns nil because HOMEBREW_BINTRAY_USER and HOMEBREW_BINTRAY_KEY are not set" do
      ENV["HOMEBREW_BINTRAY_USER"] = nil
      ENV["HOMEBREW_BINTRAY_KEY"] = nil
      expect(bottle_publisher.publish_and_check_bottles)
        .to eq nil
    end
  end

  describe "verify_bintray_published" do
    it "returns nil if no formula has been defined" do
      expect(bottle_publisher.verify_bintray_published([]))
        .to eq nil
    end

    it "fails if HOMEBREW_DISABLE_LOAD_FORMULA is set to 1" do
      ENV["HOMEBREW_DISABLE_LOAD_FORMULA"] = "1"
      stub_formula_loader(formula("foo") { url "foo-1.0" })
      expect { bottle_publisher.verify_bintray_published(["foo"]) }
        .to raise_error("Need to load formulae to verify their publication!")
    end

    it "checks if a bottle has been published" do
      stub_formula_loader(formula("foo") { url "foo-1.0" })
      expect { bottle_publisher.verify_bintray_published(["foo"]) }
        .to output("Warning: Cannot publish bottle: Failed reading info for formula foo\n").to_stderr
    end
  end
end
