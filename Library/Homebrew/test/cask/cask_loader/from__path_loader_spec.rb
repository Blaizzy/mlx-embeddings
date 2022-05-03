# typed: false
# frozen_string_literal: true

describe Cask::CaskLoader::FromPathLoader do
  describe "#load" do
    context "when the file does not contain a cask" do
      let(:path) {
        (mktmpdir/"cask.rb").tap do |path|
          path.write <<~RUBY
            true
          RUBY
        end
      }

      it "raises an error" do
        expect {
          described_class.new(path).load(config: nil)
        }.to raise_error(Cask::CaskUnreadableError, /does not contain a cask/)
      end
    end

    context "when the file calls a non-existent method" do
      let(:path) {
        (mktmpdir/"cask.rb").tap do |path|
          path.write <<~RUBY
            this_method_does_not_exist
          RUBY
        end
      }

      it "raises an error" do
        expect {
          described_class.new(path).load(config: nil)
        }.to raise_error(Cask::CaskUnreadableError, /undefined local variable or method/)
      end
    end

    context "when the file contains an outdated cask" do
      it "raises an error" do
        expect {
          described_class.new(cask_path("invalid/invalid-depends-on-macos-bad-release")).load(config: nil)
        }.to raise_error(Cask::CaskInvalidError,
                         /invalid 'depends_on macos' value: unknown or unsupported macOS version:/)
      end
    end
  end
end
