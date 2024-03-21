# frozen_string_literal: true

require "abstract_command"

RSpec.describe Homebrew::AbstractCommand do
  describe "subclasses" do
    before do
      cat = Class.new(described_class) do
        cmd_args do
          switch "--foo"
          flag "--bar="
        end
        def run; end
      end
      stub_const("Cat", cat)
    end

    describe "parsing args" do
      it "parses valid args" do
        expect { Cat.new(["--foo"]).run }.not_to raise_error
      end

      it "allows access to args" do
        expect(Cat.new(["--bar", "baz"]).args[:bar]).to eq("baz")
      end

      it "raises on invalid args" do
        expect { Cat.new(["--bat"]) }.to raise_error(OptionParser::InvalidOption)
      end
    end

    describe "command names" do
      it "has a default command name" do
        expect(Cat.command_name).to eq("cat")
      end

      it "can lookup command" do
        expect(described_class.command("cat")).to be(Cat)
      end

      describe "when command name is overridden" do
        before do
          tac = Class.new(described_class) do
            def self.command_name = "t-a-c"
            def run; end
          end
          stub_const("Tac", tac)
        end

        it "can be looked up by command name" do
          expect(described_class.command("t-a-c")).to be(Tac)
        end
      end
    end
  end

  describe "command paths" do
    it "match command name" do
      # Ensure all commands are loaded
      ["cmd", "dev-cmd"].each do |dir|
        Dir[File.join(__dir__, "../#{dir}", "*.rb")].each { require(_1) }
      end
      test_classes = ["Cat", "Tac"]

      described_class.subclasses.each do |klass|
        next if test_classes.include?(klass.name)

        dir = klass.name.start_with?("Homebrew::DevCmd") ? "dev-cmd" : "cmd"
        expect(Pathname(File.join(__dir__, "../#{dir}/#{klass.command_name}.rb"))).to exist
      end
    end
  end
end
