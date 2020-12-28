# typed: false
# frozen_string_literal: true

require "cmd/shared_examples/args_parse"
require "dev-cmd/bottle"

describe "Homebrew.bottle_args" do
  it_behaves_like "parseable arguments"
end

describe "brew bottle", :integration_test do
  it "builds a bottle for the given Formula" do
    # create stub patchelf
    if OS.linux?
      setup_test_formula "patchelf"
      patchelf = HOMEBREW_CELLAR/"patchelf/1.0/bin/patchelf"
      patchelf.dirname.mkpath
      patchelf.write <<~EOS
        #!/bin/sh
        exit 0
      EOS
      FileUtils.chmod "+x", patchelf
      FileUtils.ln_s patchelf, HOMEBREW_PREFIX/"bin/patchelf"
    end

    install_test_formula "testball", build_bottle: true

    # `brew bottle` should not fail with dead symlink
    # https://github.com/Homebrew/legacy-homebrew/issues/49007
    (HOMEBREW_CELLAR/"testball/0.1").cd do
      FileUtils.ln_s "not-exist", "symlink"
    end

    begin
      expect { brew "bottle", "--no-rebuild", "testball" }
        .to output(/testball--0\.1.*\.bottle\.tar\.gz/).to_stdout
        .and not_to_output.to_stderr
        .and be_a_success
    ensure
      FileUtils.rm_f Dir.glob("testball--0.1*.bottle.tar.gz")
    end
  end
end

def stub_hash(parameters)
  <<~EOS
    {
      "#{parameters[:name]}":{
         "formula":{
            "pkg_version":"#{parameters[:version]}",
            "path":"#{parameters[:path]}"
         },
         "bottle":{
            "root_url":"#{HOMEBREW_BOTTLE_DEFAULT_DOMAIN}",
            "prefix":"/usr/local",
            "cellar":"#{parameters[:cellar]}",
            "rebuild":0,
            "tags":{
               "#{parameters[:os]}":{
                  "filename":"#{parameters[:filename]}",
                  "local_filename":"#{parameters[:local_filename]}",
                  "sha256":"#{parameters[:sha256]}"
               }
            }
         },
         "bintray":{
            "package":"#{parameters[:name]}",
            "repository":"bottles"
         }
      }
    }
  EOS
end

describe Homebrew do
  subject(:homebrew) { described_class }

  let(:hello_hash_big_sur) {
    JSON.parse stub_hash(
      "name":           "hello",
      "version":        "1.0",
      "path":           "/home/hello.rb",
      "cellar":         "any_skip_relocation",
      "os":             "big_sur",
      "filename":       "hello-1.0.big_sur.bottle.tar.gz",
      "local_filename": "hello--1.0.big_sur.bottle.tar.gz",
      "sha256":         "a0af7dcbb5c83f6f3f7ecd507c2d352c1a018f894d51ad241ce8492fa598010f",
    )
  }
  let(:hello_hash_catalina) {
    JSON.parse stub_hash(
      "name":           "hello",
      "version":        "1.0",
      "path":           "/home/hello.rb",
      "cellar":         "any_skip_relocation",
      "os":             "catalina",
      "filename":       "hello-1.0.catalina.bottle.tar.gz",
      "local_filename": "hello--1.0.catalina.bottle.tar.gz",
      "sha256":         "5334dd344986e46b2aa4f0471cac7b0914bd7de7cb890a34415771788d03f2ac",
    )
  }
  let(:unzip_hash_big_sur) {
    JSON.parse stub_hash(
      "name":           "unzip",
      "version":        "2.0",
      "path":           "/home/unzip.rb",
      "cellar":         "any_skip_relocation",
      "os":             "big_sur",
      "filename":       "unzip-2.0.big_sur.bottle.tar.gz",
      "local_filename": "unzip--2.0.big_sur.bottle.tar.gz",
      "sha256":         "16cf230afdfcb6306c208d169549cf8773c831c8653d2c852315a048960d7e72",
    )
  }
  let(:unzip_hash_catalina) {
    JSON.parse stub_hash(
      "name":           "unzip",
      "version":        "2.0",
      "path":           "/home/unzip.rb",
      "cellar":         "any",
      "os":             "catalina",
      "filename":       "unzip-2.0.catalina.bottle.tar.gz",
      "local_filename": "unzip--2.0.catalina.bottle.tar.gz",
      "sha256":         "d9cc50eec8ac243148a121049c236cba06af4a0b1156ab397d0a2850aa79c137",
    )
  }

  specify "::parse_json_files" do
    Tempfile.open("hello--1.0.big_sur.bottle.json") do |f|
      f.write stub_hash(
        "name":           "hello",
        "version":        "1.0",
        "path":           "/home/hello.rb",
        "cellar":         "any_skip_relocation",
        "os":             "big_sur",
        "filename":       "hello-1.0.big_sur.bottle.tar.gz",
        "local_filename": "hello--1.0.big_sur.bottle.tar.gz",
        "sha256":         "a0af7dcbb5c83f6f3f7ecd507c2d352c1a018f894d51ad241ce8492fa598010f",
      )
      f.close
      expect(
        homebrew.parse_json_files([f.path]).first["hello"]["bottle"]["tags"]["big_sur"]["filename"],
      ).to eq("hello-1.0.big_sur.bottle.tar.gz")
    end
  end

  specify "::merge_json_files" do
    bottles_hash = homebrew.merge_json_files(
      [hello_hash_big_sur, hello_hash_catalina, unzip_hash_big_sur, unzip_hash_catalina],
    )

    hello_hash = bottles_hash["hello"]
    expect(hello_hash["bottle"]["cellar"]).to eq("any_skip_relocation")
    expect(hello_hash["bottle"]["tags"]["big_sur"]["filename"]).to eq("hello-1.0.big_sur.bottle.tar.gz")
    expect(hello_hash["bottle"]["tags"]["big_sur"]["local_filename"]).to eq("hello--1.0.big_sur.bottle.tar.gz")
    expect(hello_hash["bottle"]["tags"]["big_sur"]["sha256"]).to eq(
      "a0af7dcbb5c83f6f3f7ecd507c2d352c1a018f894d51ad241ce8492fa598010f",
    )
    expect(hello_hash["bottle"]["tags"]["catalina"]["filename"]).to eq("hello-1.0.catalina.bottle.tar.gz")
    expect(hello_hash["bottle"]["tags"]["catalina"]["local_filename"]).to eq("hello--1.0.catalina.bottle.tar.gz")
    expect(hello_hash["bottle"]["tags"]["catalina"]["sha256"]).to eq(
      "5334dd344986e46b2aa4f0471cac7b0914bd7de7cb890a34415771788d03f2ac",
    )
    unzip_hash = bottles_hash["unzip"]
    expect(unzip_hash["bottle"]["cellar"]).to eq("any")
    expect(unzip_hash["bottle"]["tags"]["big_sur"]["filename"]).to eq("unzip-2.0.big_sur.bottle.tar.gz")
    expect(unzip_hash["bottle"]["tags"]["big_sur"]["local_filename"]).to eq("unzip--2.0.big_sur.bottle.tar.gz")
    expect(unzip_hash["bottle"]["tags"]["big_sur"]["sha256"]).to eq(
      "16cf230afdfcb6306c208d169549cf8773c831c8653d2c852315a048960d7e72",
    )
    expect(unzip_hash["bottle"]["tags"]["catalina"]["filename"]).to eq("unzip-2.0.catalina.bottle.tar.gz")
    expect(unzip_hash["bottle"]["tags"]["catalina"]["local_filename"]).to eq("unzip--2.0.catalina.bottle.tar.gz")
    expect(unzip_hash["bottle"]["tags"]["catalina"]["sha256"]).to eq(
      "d9cc50eec8ac243148a121049c236cba06af4a0b1156ab397d0a2850aa79c137",
    )
  end
end

describe "brew bottle --merge", :integration_test, :needs_linux do
  let(:core_tap) { CoreTap.new }
  let(:tarball) do
    if OS.linux?
      TEST_FIXTURE_DIR/"tarballs/testball-0.1-linux.tbz"
    else
      TEST_FIXTURE_DIR/"tarballs/testball-0.1.tbz"
    end
  end

  before do
    Pathname("#{TEST_TMPDIR}/testball-1.0.big_sur.bottle.json").write stub_hash(
      "name":           "testball",
      "version":        "1.0",
      "path":           "#{core_tap.path}/Formula/testball.rb",
      "cellar":         "any_skip_relocation",
      "os":             "big_sur",
      "filename":       "hello-1.0.big_sur.bottle.tar.gz",
      "local_filename": "hello--1.0.big_sur.bottle.tar.gz",
      "sha256":         "a0af7dcbb5c83f6f3f7ecd507c2d352c1a018f894d51ad241ce8492fa598010f",
    )

    Pathname("#{TEST_TMPDIR}/testball-1.0.catalina.bottle.json").write stub_hash(
      "name":           "testball",
      "version":        "1.0",
      "path":           "#{core_tap.path}/Formula/testball.rb",
      "cellar":         "any_skip_relocation",
      "os":             "catalina",
      "filename":       "testball-1.0.catalina.bottle.tar.gz",
      "local_filename": "testball--1.0.catalina.bottle.tar.gz",
      "sha256":         "5334dd344986e46b2aa4f0471cac7b0914bd7de7cb890a34415771788d03f2ac",
    )
  end

  after do
    FileUtils.rm_f "#{TEST_TMPDIR}/testball-1.0.catalina.bottle.json"
    FileUtils.rm_f "#{TEST_TMPDIR}/testball-1.0.big_sur.bottle.json"
  end

  it "adds the bottle block to a formula that has none" do
    core_tap.path.cd do
      system "git", "init"
      setup_test_formula "testball"
      system "git", "add", "--all"
      system "git", "commit", "-m", "testball 0.1"
    end

    expect {
      brew "bottle",
           "--merge",
           "--write",
           "#{TEST_TMPDIR}/testball-1.0.big_sur.bottle.json",
           "#{TEST_TMPDIR}/testball-1.0.catalina.bottle.json"
    }.to output(<<~EOS).to_stdout
      ==> testball
        bottle do
          root_url "#{HOMEBREW_BOTTLE_DEFAULT_DOMAIN}"
          cellar :any_skip_relocation
          sha256 "a0af7dcbb5c83f6f3f7ecd507c2d352c1a018f894d51ad241ce8492fa598010f" => :big_sur
          sha256 "5334dd344986e46b2aa4f0471cac7b0914bd7de7cb890a34415771788d03f2ac" => :catalina
        end
    EOS

    expect((core_tap.path/"Formula/testball.rb").read).to eq <<~EOS
      class Testball < Formula
        desc "Some test"
        homepage "https://brew.sh/testball"
        url "file://#{tarball}"
        sha256 "#{tarball.sha256}"

        bottle do
          root_url "#{HOMEBREW_BOTTLE_DEFAULT_DOMAIN}"
          cellar :any_skip_relocation
          sha256 "a0af7dcbb5c83f6f3f7ecd507c2d352c1a018f894d51ad241ce8492fa598010f" => :big_sur
          sha256 "5334dd344986e46b2aa4f0471cac7b0914bd7de7cb890a34415771788d03f2ac" => :catalina
        end

        option "with-foo", "Build with foo"

        def install
          (prefix/"foo"/"test").write("test") if build.with? "foo"
          prefix.install Dir["*"]
          (buildpath/"test.c").write \
          "#include <stdio.h>\\nint main(){printf(\\"test\\");return 0;}"
          bin.mkpath
          system ENV.cc, "test.c", "-o", bin/"test"
        end



        # something here

      end
    EOS
  end

  it "replaces the bottle block in a formula that already has a bottle block" do
    core_tap.path.cd do
      system "git", "init"
      setup_test_formula "testball", bottle_block: <<~EOS

        bottle do
          root_url "#{HOMEBREW_BOTTLE_DEFAULT_DOMAIN}"
          cellar :any_skip_relocation
          sha256 "6b276491297d4052538bd2fd22d5129389f27d90a98f831987236a5b90511b98" => :big_sur
          sha256 "16cf230afdfcb6306c208d169549cf8773c831c8653d2c852315a048960d7e72" => :catalina
        end
      EOS
      system "git", "add", "--all"
      system "git", "commit", "-m", "testball 0.1"
    end

    expect {
      brew "bottle",
           "--merge",
           "--write",
           "#{TEST_TMPDIR}/testball-1.0.big_sur.bottle.json",
           "#{TEST_TMPDIR}/testball-1.0.catalina.bottle.json"
    }.to output(<<~EOS).to_stdout
      ==> testball
        bottle do
          root_url "#{HOMEBREW_BOTTLE_DEFAULT_DOMAIN}"
          cellar :any_skip_relocation
          sha256 "a0af7dcbb5c83f6f3f7ecd507c2d352c1a018f894d51ad241ce8492fa598010f" => :big_sur
          sha256 "5334dd344986e46b2aa4f0471cac7b0914bd7de7cb890a34415771788d03f2ac" => :catalina
        end
    EOS

    expect((core_tap.path/"Formula/testball.rb").read).to eq <<~EOS
      class Testball < Formula
        desc "Some test"
        homepage "https://brew.sh/testball"
        url "file://#{tarball}"
        sha256 "#{tarball.sha256}"

        option "with-foo", "Build with foo"

        bottle do
          root_url "#{HOMEBREW_BOTTLE_DEFAULT_DOMAIN}"
          cellar :any_skip_relocation
          sha256 "a0af7dcbb5c83f6f3f7ecd507c2d352c1a018f894d51ad241ce8492fa598010f" => :big_sur
          sha256 "5334dd344986e46b2aa4f0471cac7b0914bd7de7cb890a34415771788d03f2ac" => :catalina
        end

        def install
          (prefix/"foo"/"test").write("test") if build.with? "foo"
          prefix.install Dir["*"]
          (buildpath/"test.c").write \
          "#include <stdio.h>\\nint main(){printf(\\"test\\");return 0;}"
          bin.mkpath
          system ENV.cc, "test.c", "-o", bin/"test"
        end



        # something here

      end
    EOS
  end

  it "fails to add the bottle block to a formula that has no bottle block when using --keep-old" do
    core_tap.path.cd do
      system "git", "init"
      setup_test_formula("testball")
      system "git", "add", "--all"
      system "git", "commit", "-m", "testball 0.1"
    end

    expect {
      brew "bottle",
           "--merge",
           "--write",
           "--keep-old",
           "#{TEST_TMPDIR}/testball-1.0.big_sur.bottle.json",
           "#{TEST_TMPDIR}/testball-1.0.catalina.bottle.json"
    }.to output("Error: --keep-old was passed but there was no existing bottle block!\n").to_stderr
  end

  it "updates the bottle block in a formula that already has a bottle block when using --keep-old" do
    core_tap.path.cd do
      system "git", "init"
      setup_test_formula "testball", bottle_block: <<~EOS

        bottle do
          root_url "#{HOMEBREW_BOTTLE_DEFAULT_DOMAIN}"
          cellar :any_skip_relocation
          sha256 "6971b6eebf4c00eaaed72a1104a49be63861eabc95d679a0c84040398e320059" => :high_sierra
        end
      EOS
      system "git", "add", "--all"
      system "git", "commit", "-m", "testball 0.1"
    end

    expect {
      brew "bottle",
           "--merge",
           "--write",
           "--keep-old",
           "#{TEST_TMPDIR}/testball-1.0.big_sur.bottle.json",
           "#{TEST_TMPDIR}/testball-1.0.catalina.bottle.json"
    }.to output(<<~EOS).to_stdout
      ==> testball
        bottle do
          root_url "#{HOMEBREW_BOTTLE_DEFAULT_DOMAIN}"
          cellar :any_skip_relocation
          sha256 "a0af7dcbb5c83f6f3f7ecd507c2d352c1a018f894d51ad241ce8492fa598010f" => :big_sur
          sha256 "5334dd344986e46b2aa4f0471cac7b0914bd7de7cb890a34415771788d03f2ac" => :catalina
          sha256 "6971b6eebf4c00eaaed72a1104a49be63861eabc95d679a0c84040398e320059" => :high_sierra
        end
    EOS

    expect((core_tap.path/"Formula/testball.rb").read).to eq <<~EOS
      class Testball < Formula
        desc "Some test"
        homepage "https://brew.sh/testball"
        url "file://#{tarball}"
        sha256 "#{tarball.sha256}"

        option "with-foo", "Build with foo"

        bottle do
          root_url "#{HOMEBREW_BOTTLE_DEFAULT_DOMAIN}"
          cellar :any_skip_relocation
          sha256 "a0af7dcbb5c83f6f3f7ecd507c2d352c1a018f894d51ad241ce8492fa598010f" => :big_sur
          sha256 "5334dd344986e46b2aa4f0471cac7b0914bd7de7cb890a34415771788d03f2ac" => :catalina
          sha256 "6971b6eebf4c00eaaed72a1104a49be63861eabc95d679a0c84040398e320059" => :high_sierra
        end

        def install
          (prefix/"foo"/"test").write("test") if build.with? "foo"
          prefix.install Dir["*"]
          (buildpath/"test.c").write \
          "#include <stdio.h>\\nint main(){printf(\\"test\\");return 0;}"
          bin.mkpath
          system ENV.cc, "test.c", "-o", bin/"test"
        end



        # something here

      end
    EOS
  end
end
