# frozen_string_literal: true

require "system_command"
require "utils/copy"

RSpec.describe Utils::Copy do
  let(:path) { Pathname(Dir.mktmpdir) }
  let(:source) { path/"source" }
  let(:target) { path/"target" }

  RSpec.shared_examples "copies files" do |method_name|
    context "when the source is a regular file" do
      before do
        source.write "foo"
        FileUtils.touch source, mtime: 42
      end

      it "copies the file and preserves its attributes" do
        expect(target.exist?).to be(false)

        described_class.public_send(method_name, source, target)

        expect(target.file?).to be(true)
        expect(target.read).to eq(source.read)
        expect(target.mtime).to eq(source.mtime)
      end
    end

    context "when the source is a list of files and the target is a directory" do
      let(:source) { [path/"file1", path/"file2"] }
      let(:target_children) { [target/"file1", target/"file2"] }

      before do
        source.each do |source|
          source.write("foo")
          FileUtils.touch source, mtime: 42
        end
        target.mkpath
      end

      it "copies the files and preserves their attributes" do
        expect(target_children.map(&:exist?)).to all be(false)

        described_class.public_send(method_name, source, target)

        expect(target_children.map(&:file?)).to all be(true)
        target_children.zip(source) do |target, source|
          expect(target.read).to eq(source.read)
          expect(target.mtime).to eq(source.mtime)
        end
      end
    end
  end

  RSpec.shared_context "with macOS version" do |version|
    before do
      allow(MacOS).to receive(:version).and_return(MacOSVersion.new(version))
    end
  end

  RSpec.shared_examples ".*with_attributes" do |method_name, fileutils_method_name|
    context "when running on macOS Sonoma or later", :needs_macos do
      include_context "with macOS version", "14"

      include_examples "copies files", method_name

      it "executes `cp` command with `-c` flag" do
        expect(SystemCommand).to receive(:run!).with(
          a_string_ending_with("cp"),
          hash_including(args: include("-c").and(end_with(source, target))),
        )

        described_class.public_send(method_name, source, target)
      end
    end

    context "when running on Linux or macOS Ventura or earlier" do
      include_context "with macOS version", "13" if OS.mac?

      include_examples "copies files", method_name

      it "uses `FileUtils.#{fileutils_method_name}`" do
        expect(SystemCommand).not_to receive(:run!)
        expect(FileUtils).to receive(fileutils_method_name).with(source, target, hash_including(preserve: true))

        described_class.public_send(method_name, source, target)
      end

      context "when `force_command` is set" do
        it "executes `cp` command without `-c` flag" do
          expect(SystemCommand).to receive(:run!).with(
            a_string_ending_with("cp"),
            hash_including(args: exclude("-c").and(end_with(source, target))),
          )

          described_class.public_send(method_name, source, target, force_command: true)
        end
      end
    end
  end

  describe ".with_attributes" do
    include_examples ".*with_attributes", :with_attributes, :cp
  end

  describe ".recursive_with_attributes" do
    RSpec.shared_examples "copies directory" do
      context "when the source is a directory" do
        before do
          FileUtils.mkpath source, mode: 0742
          (source/"child").tap do |child|
            child.write "foo"
            FileUtils.touch child, mtime: 42
          end
        end

        it "copies the directory recursively and preserves its attributes" do
          expect(target.exist?).to be(false)

          described_class.recursive_with_attributes(source, target)

          expect(target.directory?).to be(true)
          expect(target.stat.mode).to be(source.stat.mode)

          [source/"child", target/"child"].tap do |source, target|
            expect(target.file?).to be(true)
            expect(target.read).to eq(source.read)
            expect(target.mtime).to eq(source.mtime)
          end
        end
      end
    end

    include_examples ".*with_attributes", :recursive_with_attributes, :cp_r

    context "when running on macOS Sonoma or later", :needs_macos do
      include_context "with macOS version", "14"
      include_examples "copies directory"
    end

    context "when running on Linux or macOS Ventura or earlier" do
      include_context "with macOS version", "13" if OS.mac?
      include_examples "copies directory"
    end
  end
end
