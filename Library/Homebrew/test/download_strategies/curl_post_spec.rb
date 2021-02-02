# typed: false
# frozen_string_literal: true

require "download_strategy"

describe CurlPostDownloadStrategy do
  subject(:strategy) { described_class.new(url, name, version, **specs) }

  let(:name) { "foo" }
  let(:url) { "https://example.com/foo.tar.gz" }
  let(:version) { "1.2.3" }
  let(:specs) { {} }

  describe "#fetch" do
    before do
      strategy.temporary_path.dirname.mkpath
      FileUtils.touch strategy.temporary_path
    end

    context "with :using and :data specified" do
      let(:specs) {
        {
          using: :post,
          data:  {
            form: "data",
            is:   "good",
          },
        }
      }

      it "adds the appropriate curl args" do
        expect(strategy).to receive(:system_command).with(
          /curl/,
          hash_including(args: array_including_cons("-d", "form=data").and(array_including_cons("-d", "is=good"))),
        )
        .at_least(:once)
        .and_return(instance_double(SystemCommand::Result, success?: true, stdout: "", assert_success!: nil))

        strategy.fetch
      end
    end

    context "with :using but no :data" do
      let(:specs) { { using: :post } }

      it "adds the appropriate curl args" do
        expect(strategy).to receive(:system_command).with(
          /curl/,
          hash_including(args: array_including_cons("-X", "POST")),
        )
        .at_least(:once)
        .and_return(instance_double(SystemCommand::Result, success?: true, stdout: "", assert_success!: nil))

        strategy.fetch
      end
    end
  end
end
