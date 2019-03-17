describe "brew extract", :integration_test do
  it "retrieves the specified version of formula from core tap, defaulting to most recent" do
    path = Tap::TAP_DIRECTORY/"homebrew/homebrew-foo"
    (path/"Formula").mkpath
    target = Tap.from_path(path)
    core_tap = CoreTap.new
    core_tap.path.cd do
      system "git", "init"
      formula_file = setup_test_formula "testball"
      system "git", "add", "--all"
      system "git", "commit", "-m", "testball 0.1"
      contents = File.read(formula_file)
      contents.gsub!("testball-0.1", "testball-0.2")
      File.write(formula_file, contents)
      system "git", "add", "--all"
      system "git", "commit", "-m", "testball 0.2"
    end
    expect { brew "extract", "testball", target.name }
      .to be_a_success

    expect(path/"Formula/testball@0.2.rb").to exist

    expect(Formulary.factory(path/"Formula/testball@0.2.rb").version).to be == "0.2"

    expect { brew "extract", "testball", target.name, "--version=0.1" }
      .to be_a_success

    expect(path/"Formula/testball@0.1.rb").to exist

    expect(Formulary.factory(path/"Formula/testball@0.1.rb").version).to be == "0.1"
  end

  it "retrieves the specified version of formula from a tap other than core, defaulting to most recent" do
    destination = Tap::TAP_DIRECTORY/"homebrew/homebrew-foo"
    (destination/"Formula").mkpath
    destination_tap = Tap.from_path(destination)

    source = Tap::TAP_DIRECTORY/"homebrew/homebrew-bar"
    source.mkpath
    source_tap = Tap.from_path(source)

    tarball = if OS.linux?
      TEST_FIXTURE_DIR/"tarballs/testball-0.1-linux.tbz"
    else
      TEST_FIXTURE_DIR/"tarballs/testball-0.1.tbz"
    end

    content = <<~RUBY
      desc "Some test"
      homepage "https://brew.sh/testball"
      url "file://#{tarball}"
      sha256 "#{tarball.sha256}"

      option "with-foo", "Build with foo"

      def install
        (prefix/"foo"/"test").write("test") if build.with? "foo"
        prefix.install Dir["*"]
        (buildpath/"test.c").write \
          "#include <stdio.h>\\nint main(){return printf(\\"test\\");}"
        bin.mkpath
        system ENV.cc, "test.c", "-o", bin/"test"
      end
    RUBY

    formula_file = source_tap.path/"testball.rb"
    formula_file.write <<~RUBY
      class Testball < Formula
        #{content}
      end
    RUBY

    source_tap.path.cd do
      system "git", "init"
      system "git", "add", "--all"
      system "git", "commit", "-m", "testball 0.1"
      contents = File.read(formula_file)
      contents.gsub!("testball-0.1", "testball-0.2")
      File.write(formula_file, contents)
      system "git", "add", "--all"
      system "git", "commit", "-m", "testball 0.2"
    end
    expect { brew "extract", "homebrew/bar/testball", destination_tap.name }
      .to be_a_success

    expect(destination/"Formula/testball@0.2.rb").to exist

    expect(Formulary.factory(destination/"Formula/testball@0.2.rb").version).to be == "0.2"

    expect { brew "extract", "homebrew/bar/testball", destination_tap.name, "--version=0.1" }
      .to be_a_success

    expect(destination/"Formula/testball@0.1.rb").to exist

    expect(Formulary.factory(destination/"Formula/testball@0.1.rb").version).to be == "0.1"
  end
end
